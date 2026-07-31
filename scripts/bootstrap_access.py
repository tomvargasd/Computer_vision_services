import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database import add_account, init_db, get_conn, SECRET_KEY_SETTING

init_db()

email = sys.argv[1] if len(sys.argv) > 1 else None
if not email:
    print("Uso: python scripts/bootstrap_access.py <correo>")
    print("Ej: python scripts/bootstrap_access.py admin@ejemplo.com")
    sys.exit(1)

import secrets
password = secrets.token_hex(8)

try:
    account = add_account(email, password, "full")
    print("=" * 50)
    print("  CUENTA DE ACCESO TOTAL CREADA")
    print("=" * 50)
    print(f"  Correo: {email}")
    print(f"  Clave:  {password}")
    print("=" * 50)
    print("  GUARDA ESTA CLAVE. Solo se muestra una vez.")
    print("=" * 50)
except Exception as e:
    print(f"Error: {e}")
    print("El correo ya podría estar registrado.")
    sys.exit(1)
