#from orderly.run import orderly_run
import orderly
from main import main

if __name__ == "__main__":
    orderly.dependency(None, "latest(name == 'data')",
                    {"data.pkl" : "data.pkl"}) 
    main()