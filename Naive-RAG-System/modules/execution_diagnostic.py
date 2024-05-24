import time

def execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func()
        end_time = time.time()

        execution_time = round(end_time - start_time, 1)

        print()
        print()
        print(f"Execution Time: {execution_time} seconds")
        print()
        print()
        
    return wrapper