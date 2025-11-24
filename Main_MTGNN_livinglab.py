from Main_MTGNN import run_pipeline
import time

if __name__ == "__main__":
    config_name = "Livinglab_test.yaml"
    db_name = 'Mongo_db'
    course = [1, 1, 1]
    
    tic=time.time()
    run_pipeline(mode="predict",version='v0.0.0',config_name=config_name,course=course,db_name=db_name)
    toc=time.time()
    print(f"Time taken: {toc-tic} seconds")
