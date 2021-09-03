from pokemon_ml_design.synthesize_data import synthesize_data
from pokemon_ml_design.model import IPWModel, RuleBasedModel, DeterministicRuleBasedModel
from pokemon_ml_design.evaluate import evaluate

def main():
    print('synthesize data')
    training_data, validation_data = synthesize_data()

    models = dict(IPW=IPWModel(), RULE_BASED=DeterministicRuleBasedModel())
    action_choices = dict()
    for model_name, model in models.items():
        print(f'train {model_name}')
        model.fit(data=training_data)

        print(f'predict {model_name}')
        action_choice = model.predict(validation_data['context'])

        action_choices[model_name] = action_choice

    print('evaluate')
    evaluate(validation_data, action_choices=action_choices)


if __name__ == '__main__':
    main()
