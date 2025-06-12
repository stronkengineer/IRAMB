import streamlit as st
import bcrypt
import pymongo
import os
from dotenv import load_dotenv

# --- Load environment variables and connect to MongoDB ---
load_dotenv(".env")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
users = db["users"]

# --- Language selection and translations ---
language = st.sidebar.selectbox("🌐 Language", ["English", "العربية"])
is_ar = language == "العربية"

T = {
    "title": "🔐 تسجيل الدخول / إنشاء حساب" if is_ar else "🔐 Login / Sign Up",
    "login_tab": "تسجيل الدخول" if is_ar else "Login",
    "signup_tab": "إنشاء حساب" if is_ar else "Sign Up",
    "username": "اسم المستخدم" if is_ar else "Username",
    "password": "كلمة المرور" if is_ar else "Password",
    "new_username": "اسم مستخدم جديد" if is_ar else "New Username",
    "new_password": "كلمة مرور جديدة" if is_ar else "New Password",
    "login_success": "✅ تم تسجيل الدخول! استخدم الشريط الجانبي للتنقل." if is_ar else "✅ Logged in! Use the sidebar to navigate.",
    "login_fail": "❌ بيانات اعتماد غير صحيحة." if is_ar else "❌ Invalid credentials.",
    "signup_success": "✅ تم إنشاء الحساب. الرجاء تسجيل الدخول." if is_ar else "✅ Account created. Please log in.",
    "signup_fail": "⚠️ اسم المستخدم موجود بالفعل." if is_ar else "⚠️ Username already exists.",
    "welcome": f"مرحبًا، {st.session_state.get('username', '')}!" if is_ar else f"Welcome, {st.session_state.get('username', '')}!",
    "sidebar_info": "استخدم الشريط الجانبي للوصول إلى التطبيق وصفحة الرسوم البيانية." if is_ar else "Use the sidebar to access the app and chart pages.",
    "logout": "تسجيل الخروج" if is_ar else "Logout"
}

# --- Session state initialization ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- Authentication logic ---
def login(username, password):
    user = users.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        st.session_state.logged_in = True
        st.session_state.username = username
        return True
    return False

def signup(username, password):
    if users.find_one({"username": username}):
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({"username": username, "password": hashed_pw})
    return True

# --- UI ---
st.title(T["title"])

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs([T["login_tab"], T["signup_tab"]])

    with tab1:
        username = st.text_input(T["username"], key="login_username")
        password = st.text_input(T["password"], type="password", key="login_password")
        if st.button(T["login_tab"]):
            if login(username, password):
                st.success(T["login_success"])
            else:
                st.error(T["login_fail"])

    with tab2:
        new_user = st.text_input(T["new_username"], key="signup_username")
        new_pass = st.text_input(T["new_password"], type="password", key="signup_password")
        if st.button(T["signup_tab"]):
            if signup(new_user, new_pass):
                st.success(T["signup_success"])
            else:
                st.warning(T["signup_fail"])
else:
    st.success(T["welcome"])
    st.info(T["sidebar_info"])
    if st.button(T["logout"]):
        st.session_state.logged_in = False
        st.session_state.username = ""