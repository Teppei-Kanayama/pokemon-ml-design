from pokemon_ml_design.synthesize_data import synthesize_data
from pokemon_ml_design.model import IPWModel, RuleBasedModel
from pokemon_ml_design.evaluate import evaluate

def main():
    print('load')
    training_data, validation_data = synthesize_data()

    # TODO: モデルを追加しやすくする
    print('train')
    ipw_model = IPWModel()
    rule_based_model = RuleBasedModel()

    ipw_model.fit(data=training_data)
    rule_based_model.fit(data=training_data)

    print('evaluate')
    action_choice_by_ipw_model = ipw_model.predict(validation_data['context'])
    action_choice_by_rule_based_model = rule_based_model.predict(validation_data['context'])

    evaluate(validation_data, action_choices=dict(IPW=action_choice_by_ipw_model, RULEBASED=action_choice_by_rule_based_model))

if __name__ == '__main__':
    main()
