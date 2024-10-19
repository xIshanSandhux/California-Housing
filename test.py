# XOR two bytearrays (copied from sample code provided on brightspace)
def xor(first, second):
    return bytearray(x ^ y for x, y in zip(first, second))


P1 = "This is a known message!"
C1 = "a469b1c502c1cab966965e50425438e1bb1b5f9037a4c159"  
C2 = "bf73bcd3509299d566c35b5d450337e1bb175f903fafc159" 

# Convert ASCII string P1 to bytearray (copied from sample code provided on brightspace)
P1_bytes = bytes(P1, 'utf-8')

# Convert hex strings C1 and C2 to bytearrays (copied from sample code provided on brightspace)
C1_bytes = bytearray.fromhex(C1)
C2_bytes = bytearray.fromhex(C2)


iv_encryption = xor(P1_bytes, C1_bytes)

P2_bytes = xor(C2_bytes, iv_encryption)

P2_text = P2_bytes.decode()

# recovered text
print(f"P2 text: {P2_text}")
