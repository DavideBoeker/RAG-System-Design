import time

def execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func()
        end_time = time.time()

        execution_time = round(end_time - start_time, 1)
        execution_time_minutes = execution_time / 60

        print()
        print()
        print(f"Execution Time: {execution_time} seconds or {execution_time_minutes} minutes")
        print()
        print()
        
    return wrapper