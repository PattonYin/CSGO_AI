import time
import threading

def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    print(f'Done Sleeping {seconds}...')
        
def main_1():
    
    start = time.perf_counter()
    threads = []
    for _ in range(10):
        t = threading.Thread(target=do_something, args=[1.5])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()
        
    finish = time.perf_counter()

    print(f'Finished in {round(finish-start, 2)} second(s)')

import concurrent.futures

def main_2():
    
    start = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        secs = [5, 4, 3, 2, 1]
        result = [executor.submit(do_something, sec) for sec in secs]
        for f in concurrent.futures.as_completed(result):
            print(f.result())
    
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
    
    
if __name__ == "__main__":
    # main_1()
    main_2()