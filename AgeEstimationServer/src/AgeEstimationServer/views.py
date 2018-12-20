# -*- coding:utf-8 -*-
from django.http import HttpResponse
from . import services
import json


def hello(request):
    """
    test
    """
    return HttpResponse("Hello Age Gender Estimation!")


def detect(request):
    response_dict = {'successful': False}  # handle illegal cases
    if request.method == "POST":
        params = json.loads(request.body)
        if 'photo_path' not in params:
            response_dict['message'] = 'No photo path input'
            response_dict['code'] = 1
        elif 'config' not in params:
            photo_path = params['photo_path']
            response_dict = services.detect_gender_age(photo_path)
        else:
            photo_path = params['photo_path']
            config = params['config']
            response_dict = services.detect_gender_age(photo_path, config)
    else:
        response_dict['message'] = 'Request method is invalid'

    return HttpResponse(json.dumps(response_dict, ensure_ascii=False), content_type="application/json")