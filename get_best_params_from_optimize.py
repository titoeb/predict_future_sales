import pickle
import hyperopt as hp


trials = pickle.load(open("data/trials_2.p", "rb"))

min_loss = trials.best_trial["result"]["loss"]
best_params = {key: val[0] for (key, val) in trials.best_trial["misc"]["vals"].items()}

print(f"The best loss was {min_loss} and the used params are \n{best_params}")