import requests

class IGClient:
    def __init__(self, api_key, username, password, is_demo=False):
        self.api_key = api_key
        self.username = username
        self.password = password
        self.is_demo = is_demo
        self.base_url = "https://demo-api.ig.com/gateway/deal" if is_demo else "https://api.ig.com/gateway/deal"
        
        self.cst = None
        self.x_security_token = None
        self.account_id = None
        self.account_details = None

    def login(self):
        url = f"{self.base_url}/session"
        headers = {
            "Content-Type": "application/json",
            "X-IG-API-KEY": self.api_key,
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        payload = {
            "identifier": self.username,
            "password": self.password,
        }
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print("✅ Login successful!")
            self.cst = response.headers.get("CST")
            self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
            return True
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    def fetch_accounts(self):
        url = f"{self.base_url}/accounts"
        headers = {
            "X-IG-API-KEY": self.api_key,
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token,
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            self.account_details = response.json()
            print("✅ Accounts fetched:")
            print(self.account_details)
            accounts = self.account_details.get("accounts")
            if accounts and len(accounts) > 0:
                self.account_id = accounts[0].get("accountId")
                print(f"Account ID: {self.account_id}")
                return True
            else:
                print("⚠️ No accounts found.")
                return False
        else:
            print(f"❌ Failed to fetch accounts: {response.status_code}")
            print(f"Response: {response.text}")
            return False

if __name__ == "__main__":
    API_KEY = "9e62b31c8ff6883a4c03aa8d5aca487374051821"
    USERNAME = "abudooma"  # Use your actual username/email
    PASSWORD = "Portsaid2002"  # Use your actual password
    IS_DEMO = False  # Set True if using demo environment

    client = IGClient(API_KEY, USERNAME, PASSWORD, IS_DEMO)
    if client.login():
        client.fetch_accounts()
