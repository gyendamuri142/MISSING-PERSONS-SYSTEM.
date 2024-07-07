from django.http import HttpResponse
from django.shortcuts import render,redirect
import numpy as np
from .models import* 
from django.contrib import messages
from django.core.mail import send_mail
from django.template.loader import render_to_string
from datetime import datetime
import face_recognition
import cv2, json, requests
from django.shortcuts import get_object_or_404, redirect
from django.shortcuts import render
from django.contrib.auth.models import User, auth
from django.contrib.auth.decorators import login_required
import os
from django.conf import settings
from django.utils import timezone
import pytz




# Create your views here.
def home(request):
    return render(request,"index.html")



# def detect_faces(frame):
#     # Convert the frame to grayscale for faster processing
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Perform downsampling to reduce frame size
#     resized_frame = cv2.resize(gray_frame, (0, 0), fx=0.5, fy=0.5)

#     # Define ROI for face detection (adjust as needed)
#     roi = resized_frame[100:400, 200:600]

#     # Print the dimensions of the ROI for debugging
#     print("ROI dimensions:", roi.shape)

#     # Perform face detection on the ROI
#     face_locations = face_recognition.face_locations(roi, model='hog')

#     # Print the detected face locations for debugging
#     # print("Detected face locations:", face_locations)

#     # Convert face locations to original scale
#     original_scale_locations = [(top * 2 + 100, right * 2 + 200, bottom * 2 + 100, left * 2 + 200) for (top, right, bottom, left) in face_locations]

#     return original_scale_locations



# def detect(request):
    # Initialize the camera outside of the loop
    video_capture = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize a flag to track if a face has been detected in the current video stream
    face_detected = False
    
    while True:
        ret, frame = video_capture.read()

        # # Convert the frame to grayscale for faster processing
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Perform downsampling to reduce frame size
        # resized_frame = cv2.resize(gray_frame, (0, 0), fx=0.5, fy=0.5)

        # # Define ROI for face detection (adjust as needed)
        # roi = resized_frame[100:400, 200:600]
            
        # Perform face detection on the ROI
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        # Compare detected faces with stored face images
        for face_location in face_locations:
            # Extract face ROI
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]

            # Compare face with stored face images
            for person in MissingPerson.objects.all():
                stored_image = face_recognition.load_image_file(person.image.path)
                stored_face_encoding = face_recognition.face_encodings(stored_image)[0]

                # Compare face encodings using a tolerance value
                face_encoding = face_recognition.face_encodings(face_image)
                # print(face_encoding)
                if len(face_encoding) > 0:
                    match = face_recognition.compare_faces([stored_face_encoding], face_encoding[0])
                    if match[0]:
                        name = person.first_name + " " + person.last_name
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                        
                        # Check if a face has already been detected in this video stream
                        if not face_detected:
                            print("Hi " + name + " is found")
                            
                            current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
                            subject = 'Missing Person Found'
                            from_email = ''
                            recipientmail = person.email
                            if request.method == 'POST':
                                data = json.loads(request.body)
                                latitude = data.get('latitude')
                                longitude = data.get('longitude')

                                url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}'
                                response = requests.get(url)
                                data = response.json()
                                city = data['display_name']

                            context = {"first_name": person.first_name, "last_name": person.last_name,
                                        'father_name': person.father_name, "aadhar_number": person.aadhar_number,
                                        "missing_from": person.missing_from, "date_time": current_time,
                                        "location": city}

                            html_message = render_to_string('findemail.html', context=context)
                            send_mail(subject, '', from_email, [recipientmail], fail_silently=False,
                                    html_message=html_message)
                            face_detected = True  # Set the flag to True to indicate a face has been detected
                            break  # Break the loop once a match is found
                        
        # Display the resulting image
        cv2.imshow('Camera Feed', frame)

        # Check for any key press to break the loop
        key = cv2.waitKey(1)
        if key == ord(' '):
            break

    # Release the camera and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()
    
    return render(request, "surveillance.html")





from PIL import Image, ImageDraw, ImageFont

def detect_img(request):
    if request.method == 'POST' and request.FILES['detect_img']:
        uploaded_image = request.FILES['detect_img']
        
        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Find face locations and encodings in the uploaded image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Create a PIL image object from the numpy array
        pil_image = Image.fromarray(image)
        
        # Create a draw object to draw on the PIL image
        draw = ImageDraw.Draw(pil_image)

        # Define a larger font size
        font_size = 40
        font = ImageFont.truetype("arial.ttf", font_size)
        face_detected=False

        # Iterate over each detected face in the uploaded image
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare detected face with stored face images
            for person in MissingPerson.objects.all():
                stored_image = face_recognition.load_image_file(person.image.path)
                stored_face_encoding = face_recognition.face_encodings(stored_image)[0]

                # Compare face encodings using a tolerance value
                matches = face_recognition.compare_faces([stored_face_encoding], face_encoding)
                if any(matches):
                    name = person.first_name + " " + person.last_name
                    print(name)
                    
                    # Draw a rectangle around the detected face
                    draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)
                    draw.text((left, bottom + 10), name, fill=(255, 255, 255, 0),font=font)
                    
                    # Additional logic for sending email notification, etc.
                    current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
                    current_time_utc = timezone.now()
                    # Convert UTC time to Indian Standard Time (IST)
                    indian_time_zone = pytz.timezone('Asia/Kolkata')  # Indian Standard Time (IST)
                    current_time_ist = current_time_utc.astimezone(indian_time_zone)
                    subject = 'Missing Person Found'
                    from_email = ''
                    recipientmail = person.email
                    recipient_phone_number = '+91'+str(person.phone_number)
                    print(recipient_phone_number)
                    lat = request.GET.get('latitude')
                    lng = request.GET.get('longitude')
                    print(lat,lng)
                    url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}'
                    response = requests.get(url)
                    data = response.json()
                    print(data)
                    city = data['display_name']
                    loc=Location.objects.create(
                        missing_person=person,
                        latitude=lat,
                        longitude=lng,
                        detected_at=current_time_ist,
                        address=city
                            )
                    loc.save()
                        
                    context = {"first_name":person.first_name,"last_name":person.last_name,
                                'father_name':person.father_name,"aadhar_number":person.aadhar_number,
                                "missing_from":person.missing_from,"date_time":current_time,"location":city, "lat":lat, "lon":lng}
                        #send_wapmessage(context,current_time,wapnum)
                        # send_whatsapp_message(recipient_phone_number, context)
                    html_message = render_to_string('findemail.html',context = context)
                        # Send the email
                    send_mail(subject,'', from_email, [recipientmail], fail_silently=False, html_message=html_message)
                    face_detected=True
                    break 
            if not face_detected:
                name="Unknown"
                draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0), width=2)
                draw.text((left, bottom + 10), name, fill=(255, 255, 255, 0),font=font)
        
        # Save the changes to the PIL image
        del draw  # Ensure draw is deleted before displaying the image
        # Display the modified image
        pil_image.show()

        
    return render(request, "surveillance.html")



# def detect_img(request):
    if request.method == 'POST' and request.FILES['detect_img']:
        uploaded_image = request.FILES['detect_img']
        
        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        print(image)
        # Find face locations and encodings in the uploaded image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Iterate over each detected face in the uploaded image
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare detected face with stored face images
            for person in MissingPerson.objects.all():
                stored_image = face_recognition.load_image_file(person.image.path)
                stored_face_encoding = face_recognition.face_encodings(stored_image)[0]

                # Compare face encodings using a tolerance value
                matches = face_recognition.compare_faces([stored_face_encoding], face_encoding)
                print(matches)
                if any(matches):
                    name = person.first_name + " " + person.last_name
                    print(name)
                    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    # Additional logic for sending email notification, etc.
                    # ...
        
        # Convert the image back to bytes for rendering in HTML
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        context = {'image': img_bytes}
        return render(request, 'result_img.html', context)


        # return HttpResponse(img_bytes, content_type="image/jpeg")

    return render(request, "surveillance.html")



def detect(request):
    video_capture = cv2.VideoCapture(0)
    
    # Initialize a flag to track if a face has been detected in the current video stream
    face_detected = False
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare detected face with stored face images
            for person in MissingPerson.objects.all():
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break
                print(person.image.path)
                stored_image = face_recognition.load_image_file(person.image.path)
                stored_face_encoding = face_recognition.face_encodings(stored_image)[0]

                # Compare face encodings using a tolerance value
                #tolerance = 0.6  # Adjust this tolerance as needed
                matches = face_recognition.compare_faces([stored_face_encoding], face_encoding)

                if any(matches):
                    name = person.first_name + " " + person.last_name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    # Check if a face has already been detected in this video stream
                    if not face_detected:
                        print("Hi " + name + " is found")
                        print(person.phone_number)
                        
                        current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
                        current_time_utc = timezone.now()

                        # Convert UTC time to Indian Standard Time (IST)
                        indian_time_zone = pytz.timezone('Asia/Kolkata')  # Indian Standard Time (IST)
                        current_time_ist = current_time_utc.astimezone(indian_time_zone)
                        subject = 'Missing Person Found'
                        from_email = ''
                        recipientmail = [person.email,'tracetracker1@gmail.com']
                        recipient_phone_number = '+91'+str(person.phone_number)
                        print(recipient_phone_number)
                        lat = request.GET.get('latitude')
                        lng = request.GET.get('longitude')
                        print(lat,lng)
                        url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}'
                        response = requests.get(url)
                        data = response.json()
                        print(data)
                        city = data['display_name']

                        print(city)
                        loc=Location.objects.create(
                                missing_person=person,
                                latitude=lat,
                                longitude=lng,
                                detected_at=current_time_ist,
                                address=city
                            )
                        loc.save()
                        
                        context = {"first_name":person.first_name,"last_name":person.last_name,
                                    'father_name':person.father_name,"aadhar_number":person.aadhar_number,
                                    "missing_from":person.missing_from,"date_time":current_time,"location":city, "lat":lat, "lon":lng}
                        #send_wapmessage(context,current_time,wapnum)
                        # send_whatsapp_message(recipient_phone_number, context)
                        html_message = render_to_string('findemail.html',context = context)
                        # Send the email
                        send_mail(subject,'', from_email, recipientmail, fail_silently=False, html_message=html_message)
                        face_detected = True  # Set the flag to True to indicate a face has been detected
                        break  # Break the loop once a match is found

            # Check if no face was detected in the current frame
            if not face_detected:
                name = "Unknown"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Camera Feed', frame)

        # Hit ' ' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
            
    video_capture.release()
    cv2.destroyAllWindows()
    return render(request, "surveillance.html")

# def detect(request):
    # Initialize the camera outside of the loop
    
    video_capture = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize a flag to track if a face has been detected in the current video stream
    face_detected = False
    
    while True:
        ret, frame = video_capture.read()
        
        
        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare detected face with stored face images
            for person in MissingPerson.objects.all():
                print(person.image.path)
                stored_image = face_recognition.load_image_file(person.image.path)
                stored_face_encoding = face_recognition.face_encodings(stored_image)[0]

                # Compare face encodings using a tolerance value
                matches = face_recognition.compare_faces([stored_face_encoding], face_encoding)

                if any(matches):
                    name = person.first_name + " " + person.last_name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    Identified_person=name
                    # Check if a face has already been detected in this video stream
                    if not face_detected:
                        
                        print("Hi " + name + " is found")
                        print(person)
                        
                        current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
                        current_time_utc = timezone.now()

                        # Convert UTC time to Indian Standard Time (IST)
                        indian_time_zone = pytz.timezone('Asia/Kolkata')  # Indian Standard Time (IST)
                        current_time_ist = current_time_utc.astimezone(indian_time_zone)
                        subject = 'Missing Person Found'
                        from_email = ''
                        recipientmail = person.email, 'tracetracker1@gmail.com'
                        recipient_phone_number = '+91'+str(person.phone_number)
                        # lat = request.GET.get('latitude')
                        # lng = request.GET.get('longitude')

                        # url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}'
                        # response = requests.get(url)
                        # data = response.json()
                        # print(data)
                        # city = data['display_name']

                        if request.method == 'POST':
                            data = json.loads(request.body)
                            latitude = data.get('latitude')
                            longitude = data.get('longitude')
                            # Use latitude and longitude as needed
                            print("Latitude:", latitude)
                            print("Longitude:", longitude)

                            url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}'
                            response = requests.get(url)
                            data = response.json()
                            # print(data)
                            city = data['display_name']
                            loc=Location.objects.create(
                                missing_person=person,
                                latitude=latitude,
                                longitude=longitude,
                                detected_at=current_time_ist,
                                address=city
                            )
                            loc.save()
                        key = cv2.waitKey(1)
                        print(key)
                        if key==ord(' '):
                            break

                        context = {"first_name":person.first_name,"last_name":person.last_name,
                                    'father_name':person.father_name,"aadhar_number":person.aadhar_number,
                                    "missing_from":person.missing_from,"date_time":current_time,"location": city}

                        html_message = render_to_string('findemail.html',context = context)
                        send_mail(subject,'', from_email, [recipientmail], fail_silently=False, html_message=html_message)
                        face_detected = True  # Set the flag to True to indicate a face has been detected
                        break  # Break the loop once a match is found
                
            # Check if no face was detected in the current frame
            if not face_detected:
                name = "Unknown"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        
        # Display the resulting image
        cv2.imshow('Camera Feed', frame)

        # Hit 'q' on the keyboard to quit!
        # Check for any key press
        key = cv2.waitKey(1)
        print(key)
        if key==ord(' '):
            break

 
    
    # Release the camera and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()
    
    return render(request, "surveillance.html")

# import threading
# from django.template.loader import render_to_string
# from .models import MissingPerson

# def send_email(person, lat, lng):
    current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
    subject = 'Missing Person Found'
    from_email = ''
    recipientmail = person.email
    recipient_phone_number = '+91' + str(person.phone_number)
    
    url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}'
    response = requests.get(url)
    data = response.json()
    city = data['display_name']

    context = {
        "first_name": person.first_name,
        "last_name": person.last_name,
        'father_name': person.father_name,
        "aadhar_number": person.aadhar_number,
        "missing_from": person.missing_from,
        "date_time": current_time,
        "location": city,
        "lat": lat,
        "lon": lng
    }

    html_message = render_to_string('findemail.html', context=context)
    send_mail(subject, '', from_email, [recipientmail], fail_silently=False, html_message=html_message)

# def detect(request):
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            for person in MissingPerson.objects.all():
                stored_image = face_recognition.load_image_file(person.image.path)
                stored_face_encoding = face_recognition.face_encodings(stored_image)[0]
                matches = face_recognition.compare_faces([stored_face_encoding], face_encoding)

                if any(matches):
                    name = person.first_name + " " + person.last_name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    lat = request.GET.get('latitude')
                    lng = request.GET.get('longitude')
                    send_email_thread = threading.Thread(target=send_email, args=(person,lat,lng))
                    send_email_thread.start()

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
            
    video_capture.release()
    cv2.destroyAllWindows()
    return render(request, "surveillance.html")


 
# def detect(request):
    video_capture = cv2.VideoCapture(0)
    
    # Initialize a flag to track if a face has been detected in the current video stream
    face_detected = False
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare detected face with stored face images
            for person in MissingPerson.objects.all():
                stored_image = face_recognition.load_image_file(person.image.path)
                stored_face_encoding = face_recognition.face_encodings(stored_image)[0]

                # Compare face encodings using a tolerance value
                #tolerance = 0.6  # Adjust this tolerance as needed
                matches = face_recognition.compare_faces([stored_face_encoding], face_encoding)

                if any(matches):
                    name = person.first_name + " " + person.last_name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    # Check if a face has already been detected in this video stream
                    if not face_detected:
                        print("Hi " + name + " is found")
                        print(person.phone_number)
                        
                        current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
                        subject = 'Missing Person Found'
                        from_email = ''
                        recipientmail = person.email
                        recipient_phone_number = '+91'+str(person.phone_number)
                        print(recipient_phone_number)
                        lat = request.GET.get('latitude')
                        lng = request.GET.get('longitude')
                        print(lat,lng)
                        url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}'
                        response = requests.get(url)
                        data = response.json()
                        print(data)
                        city = data['display_name']

                        print(city)

                        
                        context = {"first_name":person.first_name,"last_name":person.last_name,
                                    'father_name':person.father_name,"aadhar_number":person.aadhar_number,
                                    "missing_from":person.missing_from,"date_time":current_time,"location":city, "lat":lat, "lon":lng}
                        #send_wapmessage(context,current_time,wapnum)
                        # send_whatsapp_message(recipient_phone_number, context)
                        html_message = render_to_string('findemail.html',context = context)
                        # Send the email
                        send_mail(subject,'', from_email, [recipientmail], fail_silently=False, html_message=html_message)
                        face_detected = True  # Set the flag to True to indicate a face has been detected
                        break  # Break the loop once a match is found

            # Check if no face was detected in the current frame
            if not face_detected:
                name = "Unknown"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Camera Feed', frame)

        # Hit ' ' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
            
    video_capture.release()
    cv2.destroyAllWindows()
    return render(request, "surveillance.html")

# def detect(request):
    video_capture = cv2.VideoCapture(0)
    
    # Initialize a flag to track if a face has been detected in the current video stream
    face_detected = False
    
    while True:
        ret, frame = video_capture.read()
        
        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare detected face with stored face images
            for person in MissingPerson.objects.all():
                stored_image = face_recognition.load_image_file(person.image.path)
                stored_face_encoding = face_recognition.face_encodings(stored_image)[0]

                # Compare face encodings using a tolerance value
                #tolerance = 0.6  # Adjust this tolerance as needed
                matches = face_recognition.compare_faces([stored_face_encoding], face_encoding)

                if any(matches):
                    name = person.first_name + " " + person.last_name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    # Check if a face has already been detected in this video stream
                    if not face_detected:
                        print("Hi " + name + " is found")
                        
                        current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
                        subject = 'Missing Person Found'
                        from_email = 'pptodo01@gmail'
                        recipientmail = person.email
                        recipient_phone_number = '+91'+str(person.phone_number)
                        print(recipient_phone_number)
                        context = {"first_name":person.first_name,"last_name":person.last_name,
                                    'fathers_name':person.father_name,"aadhar_number":person.aadhar_number,
                                    "missing_from":person.missing_from,"date_time":current_time,"location":"India"}
                        #send_wapmessage(context,current_time,wapnum)
                        
                        html_message = render_to_string('findemail.html',context = context)
                        # Send the email
                        send_mail(subject,'', from_email, [recipientmail], fail_silently=False, html_message=html_message)
                        face_detected = True  # Set the flag to True to indicate a face has been detected
                        break  # Break the loop once a match is found

            # Check if no face was detected in the current frame
            if not face_detected:
                name = "Unknown"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Camera Feed', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video_capture.release()
    cv2.destroyAllWindows()
    return render(request, "surveillance.html")

def surveillance(request):
    return render(request,"surveillance.html")

def location(request):
    locations=Location.objects.all()
    context={'locations':locations}
    return render(request,"location.html",context)

@login_required
def registercase(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        father_name = request.POST.get('father_name')
        date_of_birth = request.POST.get('date_of_birth')
        address = request.POST.get('address')
        phone_number = request.POST.get('phone_number')
        aadhar_number = request.POST.get('aadhar_number')
        missing_from = request.POST.get('missing_from')
        email = request.POST.get('email')
        image = request.FILES.get('image')
        gender = request.POST.get('gender')
        aadhar = MissingPerson.objects.filter(aadhar_number=aadhar_number)
        if aadhar.exists():
            messages.info(request, 'Aadhar Number already exists')
            return redirect('/registercase')
        person = MissingPerson.objects.create(
            first_name = first_name,
            last_name = last_name,
            father_name = father_name,
            date_of_birth = date_of_birth,
            address = address,
            phone_number = phone_number,
            aadhar_number = aadhar_number,
            missing_from = missing_from,
            email = email,
            image = image,
            gender = gender,
        )
        person.save()
        messages.success(request,'Case Registered Successfully')
        current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
        subject = 'Case Registered Successfully'
        from_email = ''
        recipientmail = person.email
        context = {"first_name":person.first_name,"last_name":person.last_name,
                    'father_name':person.father_name,"aadhar_number":person.aadhar_number,
                    "missing_from":person.missing_from,"date_time":current_time}
        html_message = render_to_string('regmail.html',context = context)
        # Send the email
        send_mail(subject,'', from_email, [recipientmail], fail_silently=False, html_message=html_message)

    return render(request,"registercase.html")


def  missing(request):
    queryset = MissingPerson.objects.all()
    search_query = request.GET.get('search', '')
    if search_query:
        queryset = queryset.filter(aadhar_number__icontains=search_query)
    
    context = {'missingperson': queryset}
    return render(request,"missing.html",context)


def delete_person(request, person_id):
    person = get_object_or_404(MissingPerson, id=person_id)
    photo_path = os.path.join(settings.MEDIA_ROOT, str(person.image))
    if os.path.exists(photo_path):
        os.remove(photo_path)
    person.delete()
    return redirect('missing')  # Redirect to the missing view after deleting


def update_person(request, person_id):
    person = get_object_or_404(MissingPerson, id=person_id)

    if request.method == 'POST':
        # Retrieve data from the form
        first_name = request.POST.get('first_name', person.first_name)
        last_name = request.POST.get('last_name', person.last_name)
        father_name = request.POST.get('father_name', person.father_name)
        date_of_birth = request.POST.get('date_of_birth', person.date_of_birth)
        address = request.POST.get('address', person.address)
        email = request.POST.get('email', person.email)
        phone_number = request.POST.get('phone_number', person.phone_number)
        aadhar_number = request.POST.get('aadhar_number', person.aadhar_number)
        missing_from = request.POST.get('missing_from', person.missing_from)
        gender = request.POST.get('gender', person.gender)

        # Check if a new image is provided
        new_image = request.FILES.get('image')
        if new_image:
            person.image = new_image

        # Update the person instance
        person.first_name = first_name
        person.last_name = last_name
        person.father_name = father_name
        person.date_of_birth = date_of_birth
        person.address = address
        person.email = email
        person.phone_number = phone_number
        person.aadhar_number = aadhar_number
        person.missing_from = missing_from
        person.gender = gender

        # Save the changes
        person.save()

        return redirect('missing')  # Redirect to the missing view after editing

    return render(request, 'edit.html', {'person': person})

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(password=password,username=username)

        if user is not None:
            auth.login(request, user)
            return redirect('/')
        else:
            messages.info(request, 'Info Invalid')
            return redirect('login')
    else:
        return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        cpassword = request.POST['cpassword']

        if password == cpassword:
            if User.objects.filter(email=email).exists():
                messages.info(request, 'Email already exists')
                return redirect('/register')
            elif User.objects.filter(username=username).exists():
                messages.info(request, 'Username already exists')
                return redirect('/register')
            else:
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save();
                return redirect('/login')
        else:
            messages.info(request, 'Passwords not same')
            return redirect('/register')
    else:
        return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('/')