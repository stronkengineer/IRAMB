import requests
import json

class IGAuthenticator:
    def __init__(self, api_key, username, password, base_url='https://demo-api.ig.com/gateway/deal'):
        self.api_key = api_key
        self.username = username
        self.password = password
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {
            'X-IG-API-KEY': self.api_key,
            'Content-Type': 'application/json; charset=UTF-8',
            'Accept': 'application/json; charset=UTF-8',
        }

    def login(self):
        payload = {'identifier': self.username, 'password': self.password}
        res = self.session.post(f'{self.base_url}/session', json=payload, headers=self.headers)
        if res.status_code == 200:
            self.headers['X-SECURITY-TOKEN'] = res.headers['X-SECURITY-TOKEN']
            self.headers['CST'] = res.headers['CST']
            print("Logged in successfully!")
            return True
        else:
            raise Exception(f"Login failed: {res.status_code} - {res.text}")

    def get_headers(self):
        return self.headers

    def get_prices(self, epic):
        """
        Get market prices for a given epic (market identifier).
        """
        url = f"{self.base_url}/prices/{epic}"
        res = self.session.get(url, headers=self.headers)
        if res.status_code == 200:
            return res.json()
        else:
            raise Exception(f"Failed to get prices: {res.status_code} - {res.text}")

    def place_order(self, epic, size, direction, order_type='MARKET', currency_code='USD', expiry='DFB', level=None):
        """
        Place an order.

        Parameters:
        - epic: market epic string (e.g. 'CS.D.EURUSD.CFD.IP')
        - size: float, number of units
        - direction: 'BUY' or 'SELL'
        - order_type: e.g. 'MARKET', 'LIMIT', 'STOP'
        - currency_code: currency of the trade
        - expiry: expiry code for the instrument (default 'DFB' = daily)
        - level: price level for LIMIT/STOP orders (optional)
        """
        url = f"{self.base_url}/positions/otc"
        order_data = {
            "epic": epic,
            "size": size,
            "direction": direction.upper(),
            "orderType": order_type,
            "currencyCode": currency_code,
            "expiry": expiry,
            "forceOpen": True,  # open a new position rather than hedge
            "guaranteedStop": False,
            "timeInForce": "FILL_OR_KILL",
            "quoteId": None,  # should ideally get this from price data
        }

        # Optional: if LIMIT or STOP, level must be provided
        if order_type in ['LIMIT', 'STOP']:
            if level is None:
                raise ValueError("Level must be provided for LIMIT or STOP orders")
            order_data["level"] = level

        # To get quoteId, we need a fresh price snapshot
        prices = self.get_prices(epic)
        try:
            order_data['quoteId'] = prices['prices'][0]['snapshotId']
        except (KeyError, IndexError):
            raise Exception("Failed to retrieve quoteId from prices")

        res = self.session.post(url, json=order_data, headers=self.headers)
        if res.status_code in (200, 201):
            return res.json()
        else:
            raise Exception(f"Failed to place order: {res.status_code} - {res.text}")
