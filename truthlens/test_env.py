try:
    import fastapi
    print("FastAPI imported successfully.")
    import uvicorn
    print("Uvicorn imported successfully.")
    import api.main
    print("api.main imported successfully.")
except Exception as e:
    print(f"Error: {e}")
