
# Neurodata Lab Summer Internship 2019

В рамках данной задачи предлагается решить проблему, возникающую при работе с зашумленным звуком. Решение предполагает построение модели, использующее стандартные ML практики или нейронные сети, ее обучение и тестирование.

## Описание

Необходимо реализовать алгоритм, позволяющий определить, является ли аудиозапись зашумленной или нет.

## Описание данных

Для обучения и тестирования модели будет предоставлена выборка mel-спектрограмм, построенным по чистым и зашумленным звуковым файлам с шолосом человека. Чистым звуком считается звук голоса без посторонних шумов (возможно с паузами). Шумом на зашумленных аудиозаписях может являться любой посторонний звук, который можно услышать в повседневной жизни – звонок телефона, проезжающая машина, чайник, смех и т.д.

В качестве данных будет предоставлено 14 тысяч пар mel-спектрограмм (чистый звук + зашумленный звук), соответствующих звуковым файлам, все файлы разных длительностей. Из них 12 тысячи – тренировочная выборка, 2 – валидационная.

Тестирование будет проводиться на закрытой выборке, состоящей из 2 тысяч спектрограмм.

Все данные разделены на чистые и зашумленные. Внутри каждой категории есть разделение по спикерам. Для каждого спикера есть набор из чистых данных и набор из зашумленных данных. Название файла с зашумленными данными соответствует названию файла с оригинальными данными.

Топология данных для тренировочной выборки (для тестовой идентична):

    clean/
    	speaker1_id/
    		spec_sample_1_1.npy
    		spec_sample_1_2.npy
    		...
    	speaker2_id/
    		spec_sample_2_1.npy
    		spec_sample_2_2.npy
    		...
		...
    noise/
    	speaker1_id/
    		spec_sample_1_1.npy
    		spec_sample_1_2.npy
    		...
    	speaker2_id/
    		spec_sample_2_1.npy
    		spec_sample_2_2.npy
    		...
		...

## Метрики

Метрика для алгоритма классификации: accuracy
