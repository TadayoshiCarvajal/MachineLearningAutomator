from file_handling import FileHandler
from sys import argv
    
if __name__ == '__main__':
    file_name = argv[1]
    print(file_name)

    model = FileHandler().load_model(file_name)

    while True:
        print('Type a line:')
        line = [[input(), None]]
        guess = model.predict(line)
        print('model guessed:', guess[0])
        continue_decision = input('type "Q" to quit or ENTER to try again: ')
        if continue_decision.lower() == 'q':
            break