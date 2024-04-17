#from orderly.run import orderly_run
import orderly
from run_post import main

if __name__ == "__main__":
    orderly.dependency(None, "latest(name == 'benchmark')",
                    {"experiment_results_final.csv" : "experiment_results_final.csv"}) 
    main()