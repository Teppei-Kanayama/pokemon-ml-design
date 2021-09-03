from pokemon_ml_design.synthesize_data import synthesize_data
from pokemon_ml_design.model import IPWModel, RuleBasedModel, DeterministicRuleBasedModel
from pokemon_ml_design.evaluate import validate, evaluate

def main():
    print('synthesize data')
    training_data, validation_data, test_data = synthesize_data()

    models = dict(IPW=IPWModel(), RULE_BASED=DeterministicRuleBasedModel())

    action_choices_validation = dict()
    action_choices_test = dict()
    for model_name, model in models.items():
        print(f'train {model_name}')
        model.fit(data=training_data)

        print(f'predict {model_name}')
        action_choices_validation[model_name] = model.predict(validation_data['context'])
        action_choices_test[model_name] = model.predict(test_data['context'])

    print('validate')
    validate(validation_data, action_choices=action_choices_validation)

    print('evaluate')
    evaluate(test_data, action_choices=action_choices_test)


if __name__ == '__main__':
    main()
