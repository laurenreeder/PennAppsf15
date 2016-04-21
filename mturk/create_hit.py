import sys
import urllib2
import urllib
import json
import hmac
import base64
import collections
import sha
from datetime import datetime

AWS_ACCESS_KEY_ID = "AKIAJKPJQGJNYMWPMR6Q"
AWS_SECRET_ACCESS_KEY = "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"
SERVICE = "AWSMechanicalTurkRequester"
CREATE_HIT_OP = "CreateHIT"

create_hit_req_str = "https://mechanicalturk.sandbox.amazonaws.com/?Service=AWSMechanicalTurkRequester" \
"&AWSAccessKeyId=AKIAJKPJQGJNYMWPMR6Q" \
"&Operation=%s" \
"&Signature=%s" \
"&Timestamp=%s" \
"&ResponseGroup.0=Minimal"

title_desc_quest = "&Title=%s&Description=%s&Question=%s"
end_str = "&Reward.1.Amount=%s&Reward.1.CurrencyCode=USD&AssignmentDurationInSeconds=60&LifetimeInSeconds=31536000"

def create_hit_request(image, labels):
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    signature = create_signature(SERVICE, CREATE_HIT_OP, timestamp)
    requesturl = create_hit_req_str%(CREATE_HIT_OP, signature, timestamp) + title_desc_quest%(get_title(), get_description(), make_question(image, labels)) + end_str%("1")
    return requesturl

def get_balance_request():
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    signature = create_signature(SERVICE, "GetAccountBalance", timestamp)
    requesturl = create_hit_req_str%("GetAccountBalance", signature, timestamp)
    return requesturl

def get_title():
    return urllib.quote("Image Classification")

def get_description():
    return urllib.quote("We are doing simple image classification as part of a machine learning project for school. To answer the question in the HIT, you will be shown an image and need to select the label that describes the image from the list we provide. If you do not know which of the labels the image is or if it is unclear you can click unknown. For example, there may be a picture of a jacket and the labels will be 'jacket' and 'pants'.")

def create_signature(service, operation, timestamp):
    my_sha_hmac = hmac.new(AWS_SECRET_ACCESS_KEY, service + operation + timestamp, sha)
    b64_encoded = urllib.quote(base64.encodestring(my_sha_hmac.digest()).strip())
    return b64_encoded

def make_question(image, labels):
    questionform = '<QuestionForm xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionForm.xsd">'
    questionform += "<Question><QuestionIdentifier>image_label</QuestionIdentifier><IsRequired>true</IsRequired>"
    questionform += "<QuestionContent><Text>Which of these labels best describes the contents of this image?</Text><Binary><MimeType><Type>image</Type></MimeType><DataURL>" + image + "</DataURL><AltText>Image to classify</AltText></Binary></QuestionContent>"
    questionform += "<AnswerSpecification><SelectionAnswer><StyleSuggestion>radiobutton</StyleSuggestion><Selections>"
    for i, l in enumerate(labels):
        questionform += "<Selection><SelectionIdentifier>" + l + "</SelectionIdentifier><Text>" + l + "</Text></Selection>"
    questionform += "<Selection><SelectionIdentifier>unknown</SelectionIdentifier><Text>None of the above / Not Sure</Text></Selection></Selections></SelectionAnswer></AnswerSpecification></Question></QuestionForm>"
    
    return urllib.quote(questionform)

def create_hit(image, labels):
    requesturl = create_hit_request(image, labels)
    print requesturl
    try:
        data = urllib2.urlopen(requesturl)
        try:
            response = data.read()
            print response
        except ValueError:
            sys.stderr.write('JSON ERROR\n')
            return None
    except urllib2.HTTPError as e:
        print e
        sys.stderr.write('BAD REQUEST\n')





