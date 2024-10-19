import socket

# Function to connect to Bob's oracle and interact with it
def connect_to_oracle():
    host = "10.9.0.80"  # IP address of the oracle
    port = 3000         # Port number of the oracle

    # Create a socket connection to the oracle
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        # Receive Bob's secret ciphertext and IV
        initial_data = s.recv(1024).decode('utf-8')
        print(initial_data)

        # Parse Bob's secret ciphertext and IV
        lines = initial_data.splitlines()
        secret_ciphertext = lines[1].split(': ')[1]
        iv_used = lines[2].split(': ')[1]

        print(f"Bob's secret ciphertext: {secret_ciphertext}")
        print(f"IV used: {iv_used}")

        # Predictable IV attack: Send your plaintext in hex and get the resulting ciphertext
        for i in range(1, 4):
            # You can craft different plaintexts for comparison
            plaintext = "11223344aabbccdd" if i == 1 else "deadbeefcafe1234"
            print(f"\nSending crafted plaintext (hex): {plaintext}")

            # Send the crafted plaintext (hex format)
            s.sendall(plaintext.encode('utf-8'))

            # Receive response from the oracle (next IV and resulting ciphertext)
            response = s.recv(1024).decode('utf-8')
            print(f"Oracle response:\n{response}")

            # Parse the next IV and resulting ciphertext
            response_lines = response.splitlines()
            next_iv = response_lines[0].split(': ')[1]
            ciphertext = response_lines[1].split(': ')[1]

            print(f"Next IV: {next_iv}")
            print(f"Resulting ciphertext: {ciphertext}")

        # Exit the connection
        print("\nExiting the oracle interaction.")
        s.close()

if __name__ == "__main__":
    connect_to_oracle()
