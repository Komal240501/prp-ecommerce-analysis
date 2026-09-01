"""
Simple username/password login gate for Streamlit deployment.

Usage in your main app file:

    from login import check_login
    if not check_login():
        st.stop()
    # ... rest of your dashboard code below ...

Credentials are stored as a username -> sha256 password hash dict below.
For production, move CREDENTIALS into st.secrets (see note at bottom)
instead of hardcoding them in the repo.
"""
import streamlit as st
import hashlib

# username: sha256(password)
CREDENTIALS = {
    "admin": hashlib.sha256("changeme123".encode()).hexdigest(),
}


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_login() -> bool:
    """Renders a login form. Returns True once the user is authenticated."""
    if st.session_state.get("authenticated"):
        return True

    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        stored_hash = CREDENTIALS.get(username)
        if stored_hash and _hash(password) == stored_hash:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    return False


def logout_button():
    if st.session_state.get("authenticated"):
        if st.sidebar.button("Log out"):
            st.session_state["authenticated"] = False
            st.rerun()

# --- Production note ---
# Instead of the hardcoded CREDENTIALS dict above, put this in
# .streamlit/secrets.toml (and add that file to .gitignore):
#
#   [credentials]
#   admin = "sha256_hash_here"
#
# Then load it with: CREDENTIALS = dict(st.secrets["credentials"])
