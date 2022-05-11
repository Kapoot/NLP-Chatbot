"""
Dylan Kapustka, Abed Ahmed
Dr. Mazidi
CS 4395.001
Homework 8 - Chatbot
11-14-2021
"""


import random
import json
import nltk
nltk.download('punkt')
import torch
import pickle
from os.path import exists
from neuralnet import NeuralNet
from process import bag_of_words, tokenize


class Person:

    def __init__(self, name):
        self.name = name
        self.badpuns = list()
        self.goodpuns = list()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r',encoding="utf-8") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Punny Bot"

print("Hey there! Welcome to the Punn-iest bot, Where I tell you "
      "the best puns! type (quit) to close and save our session!\n")

person_facts = {}
if exists('datastore'):
    with open('datastore', 'rb') as file:
        person_facts = pickle.load(file)

name = ""
current_person = None
while True:

    if name == "":
        name = input("What's your name?: ")
        if name.lower() not in person_facts:
            current_person = Person(name.lower())
            person_facts[name.lower()] = current_person
            print(f"{bot_name}: Nice to meet you," , name.title()+"! My name is Punny Bot.")
            print(f"{bot_name}: Would you like to hear a pun?")
        else:
            current_person = person_facts[name.lower()]
            print(f"{bot_name}: Welcome back,", current_person.name.title() + "!"," I see you missed me!")
            print(f"{bot_name}: Would you like to hear a pun?")
        continue

    if name != "":
        sentence = input(name.title() + ": ")
        if sentence == "quit":
            picklefile = open('datastore', 'wb')
            pickle.dump(person_facts, picklefile)
            picklefile.close()
            quit()
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"] and tag != 'pun':
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                if tag =='bad-puns':
                    if len(current_person.badpuns) == 0:
                        print(f"{bot_name}: It seems you liked all my jokes! I couldn't find any bad puns you saved")
                        print(f"{bot_name}: Would you like to hear another?")
                        break
                    else:
                        badpun = random.choice(current_person.badpuns)
                        print(badpun)
                        print(f"{bot_name}: Did I remember this one correctly?")
                        check_bad = input(name.title() + ": ")
                        while "yes" != check_bad and "no" != check_bad:
                            print(f"{bot_name}: pleasee, just tell me. Keep it simple, yes or no")
                            check_bad = input(name.title() + ": ")
                        if check_bad == "yes":
                            print(f"{bot_name}: Awesome! I am glad I remembered that bad joke for you")
                            print(f"{bot_name}: Would you like to hear another?")
                            break
                        elif check_bad == "no":
                            current_person.badpuns.remove(badpun)
                            print(f"{bot_name}: You're a liar!! Haha just kidding, I'll go ahead and remove that "
                                  "joke from the list then")
                            print(f"{bot_name}: Would you like to hear another?")
                            break
                if tag == 'favorite':
                    if len(current_person.goodpuns) == 0:
                        print(f"{bot_name}: It seems you hated all my jokes! I couldn't find any good puns you saved")
                        print(f"{bot_name}: Would you like to hear another?")
                        break
                    else:
                        goodpun = random.choice(current_person.goodpuns)
                        print(goodpun)
                        print("Did I remember this one correctly?")
                        check_good = input(name.title() + ": ")
                        while "yes" != check_good and "no" != check_good:
                            print(f"{bot_name}: pleasee, just tell me. Keep it simple, yes or no")
                            check_good = input(name.title() + ": ")
                        if check_good == "yes":
                            print(f"{bot_name}: Awesome! I am glad I remembered that good joke for you")
                            print(f"{bot_name}: Would you like to hear another?")
                            break
                        elif check_good == "no":
                            current_person.goodpuns.remove(goodpun)
                            print(f"{bot_name}: You're a liar!! Haha just kidding, I'll go ahead and remove that "
                                  "joke from the list then")
                            print(f"{bot_name}: Would you like to hear another?")
                            break
                if tag == 'pun':
                    current_pun = random.choice(intent['responses'])
                    print(f"{bot_name}: {current_pun}")
                    print(f"{bot_name}: Did you like that one?")
                    joke = input(name.title() + ": ")
                    while "yes" != joke and "no" != joke:
                        print(f"{bot_name}: pleaseee, just tell me. Keep it simple, yes or no")
                        joke = input(name.title() + ": ")
                    if joke == "yes" and current_pun not in current_person.goodpuns:
                        current_person.goodpuns.append(current_pun)
                        print(f"{bot_name}: I am glad you liked it! I'll go ahead and save that one for another time!")
                        print(f"{bot_name}: Would you like to hear another?")
                        break
                    elif joke == "no" and current_pun not in current_person.badpuns:
                        current_person.badpuns.append(current_pun)
                        print(f"{bot_name}: I am terribly sorry you have bad taste, but I will avoid "
                              f"that joke from now on")
                        print(f"{bot_name}: Would you like to hear another?")
                        break
        else:
            print(f"{bot_name}: I do not understand...")



