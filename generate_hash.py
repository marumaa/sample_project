import streamlit_authenticator as stauth
import bcrypt

# パスワードをハッシュ化
password = 'admin123'
salt = bcrypt.gensalt()
hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)

print("生成されたパスワードハッシュ:")
print(hashed_password.decode('utf-8')) 