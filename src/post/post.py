#from orderly.run import orderly_run
import orderly
from run_post import main

debug = True

if __name__ == "__main__":
    if debug == False:
        orderly.dependency(None, "latest(name == 'benchmark')",
                        {"experiment_results_final.csv" : "experiment_results_final.csv"}) 
    main()