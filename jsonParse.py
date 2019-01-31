import json

dat = '''{
    "name": "name",
    "age": 99,
    "gender": "male",
    "account": 10000,
    "address": "Republic of Korea",
    "hobby": [ "swimming", "reading" ],
    "family": { "father": "father", "mother": "mother", "brother": "brother" },
    "company": "company"
}'''

jdata = json.loads(dat)


def print_array(a):
    for value in a:
        print(value)


def print_object(o):
    for value in o:
        print(value, ":", o[value])


for i in jdata:
    if str(type(jdata[i])) == "<class 'dict'>":   # object
        print("------ %s ------" % i)
        print_object(jdata[i])
    elif str(type(jdata[i])) == "<class 'list'>":   # array
        print("------ %s ------" % i)
        print_array(jdata[i])
    else:
        print(i, ":", jdata[i])


