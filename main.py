from pokemon_ml_design.synthesize_data import synthesize_data
from pokemon_ml_design.model import IPWModel
from pokemon_ml_design.evaluate import evaluate

def main():
    training_data, validation_data = synthesize_data()
    ipw_model = IPWModel()
    ipw_model.fit(data=training_data)
    action_choice_by_ipw_learner = ipw_model.predict(validation_data['context'])
    evaluate(validation_data, action_choices=dict(IPW=action_choice_by_ipw_learner, IPW2=action_choice_by_ipw_learner))

if __name__ == '__main__':
    main()
