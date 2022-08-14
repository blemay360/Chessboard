from threading import Thread
import concurrent.futures
import time

thread_running = True


def my_forever_while():
    global thread_running

    start_time = time.time()

    # run this while there is no input
    while thread_running:
        time.sleep(0.1)

        if time.time() - start_time >= 5:
            start_time = time.time()
            print('Another 5 seconds has passed')


def take_input():
    user_input = input('Type user input: ')
    # doing something with the input
    return user_input

#with concurrent.futures.ThreadPoolExecutor() as executor:
    #future = executor.submit(take_input)
    #return_value = future.result()
    #print(return_value)

if __name__ == '__main__':
    
    t1 = Thread(target=my_forever_while)
    #t2 = Thread(target=take_input)

    t1.start()
    #t2.start()
    
    user_input = take_input()

    #t2.join()  # interpreter will wait until your process get completed or terminated
    print('User input: ' + user_input)
    thread_running = False
    print('The end')
