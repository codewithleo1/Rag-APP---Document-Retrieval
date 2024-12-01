from cryptography.fernet import Fernet

# Step 1: Generate a key and save it securely
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Save the key to a secure location (e.g., an environment variable or a key file)
with open("secret.key", "wb") as key_file:
    key_file.write(key)

# Step 2: Read the contents of the secrets.json file
with open("secrets.json", "rb") as file:
    file_data = file.read()

# Step 3: Encrypt the file data
encrypted_data = cipher_suite.encrypt(file_data)

# Step 4: Save the encrypted data to a new file
with open("secrets_encrypted.json", "wb") as encrypted_file:
    encrypted_file.write(encrypted_data)

print("secrets.json file encrypted successfully.")
