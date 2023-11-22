from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, StringField, IntegerField
from wtforms.validators import DataRequired, NumberRange

class ClassifierForm(FlaskForm):
    classifier = SelectField('Escolha o Classificador:', choices=[
        ('KNN', 'KNN'),
        ('SVM', 'SVM'),
        ('DT', 'Decision Tree'),
        ('RF', 'Random Forest')
    ])
    submit = SubmitField('Enviar')

class KNNForm(FlaskForm):
    n_neighbors = IntegerField('Vizinhos', validators=[DataRequired(), NumberRange(min=1)])
    submit = SubmitField('Treinar KNN')
    
    # Um campo para inserir o número de vizinhos para o algoritmo KNN. O campo requer um número inteiro maior que 0.
    
class SVMForm(FlaskForm):
    kernel = SelectField('Kernel', choices=[('linear', 'Linear'), ('poly', 'Polinomial')])
    degree = IntegerField('Grau', validators=[NumberRange(min=1)], default=3)
    submit = SubmitField('Treinar SVM')

    # kernel: Um campo de seleção para escolher o tipo de kernel (Linear ou Polinomial) para o SVM.
    # degree: Um campo para inserir o grau do kernel polinomial. O campo requer um número inteiro maior que 0 e tem o valor padrão de 3.
    
class DTForm(FlaskForm):
    max_depth = IntegerField('Profundidade Máxima', validators=[NumberRange(min=1)])
    submit = SubmitField('Treinar DT')
    
    # max_depth: Um campo para inserir a profundidade máxima da árvore de decisão. O campo requer um número inteiro maior que 0.

class RFForm(FlaskForm):
    n_estimators = IntegerField('Número de Árvores', validators=[DataRequired(), NumberRange(min=1)])
    max_depth = IntegerField('Profundidade Máxima', validators=[NumberRange(min=1)], default=None)
    submit = SubmitField('Treinar RF')
    
    # n_estimators: Um campo para inserir o número de árvores na Floresta Aleatória. O campo requer um número inteiro maior que 0.
    # max_depth: Um campo para inserir a profundidade máxima das árvores na Floresta Aleatória.