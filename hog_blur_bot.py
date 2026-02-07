import nest_asyncio,os
nest_asyncio.apply()

#########################################################

Bot_Token = os.getenv('TOKEN')

########################################################

from pyrogram import Client, filters,enums,StopTransmission,idle
from pyrogram.types import Message

import cv2,os,shutil
from tf_keras.preprocessing.image import img_to_array
from tf_keras.models import load_model
from tf_keras.utils import get_file
import numpy as np
import cvlib as cv


gender_model_path = '/content/gender_detection.model'
gender_model = load_model(gender_model_path)

detect_model = cv2.HOGDescriptor()
detect_model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

Api_Id = 15952578
Api_Hash = '3600ce5f8f9b9e18cba0f318fa0e3600'
Audio_Forms = (".mp3",".ogg",".m4a",".aac",".flac",".wav",".wma",".opus",".3gpp")

async def Create_Dir(Dir):
  if not os.path.isdir(Dir):
    os.makedirs(Dir, exist_ok=True)

async def Check_Dir(Dir):
  if os.path.isdir(Dir):
      shutil.rmtree(Dir)
  await Create_Dir(Dir)

async def Mp3_Conv(File):
  mainDir = '/'.join(File.split('/')[:-1]) + '/'
  Mp3_File = mainDir +  File.split('/')[-1].split('.')[0] + '_Conv.mp3'
  Mp3_Cmd = f'ffmpeg -i "{File}" -q:a 0 -map a "{Mp3_File}" -y'
  os.system(Mp3_Cmd)
  return Mp3_File

async def Media_Compress(file_path,Rate=None):
  mainDir = '/'.join(file_path.split('/')[:-1]) + '/'
  if file_path.lower().endswith(Audio_Forms) : 
   Res_File = mainDir + file_path.split('/')[-1].split('.')[0] + '_Comp.mp3'
   Comp_Cmd = f'ffmpeg -i "{file_path}" -b:a "{Rate}" "{Res_File}" -y '
  else :
    Res_File = mainDir + file_path.split('/')[-1].split('.')[0] + '_Comp.mp4'
    Comp_Cmd = f'ffmpeg -i "{file_path}" -c:v libx264 -crf 28 "{Res_File}" -y'
  os.system(Comp_Cmd)
  return Res_File

async def Vid_Mk(Vid,Aud):
  mainDir = '/'.join(Vid.split('/')[:-1]) + '/'
  Vid_Res = mainDir + Vid.split('/')[-1].split('.')[0] + '_Merged.mp4'
  Sub_Cmd = f'ffmpeg -i "{Vid}" -i "{Aud}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "{Vid_Res}" -y'
  os.system(Sub_Cmd)
  return Vid_Res

def Pyrogram_Client(Bot_Token):
  Bot_Identifier = Bot_Token.split(':')[0]
  Session_file = Bot_Identifier+'_session_prm_bot'
  bot = Client(Session_file,api_id=Api_Id,api_hash=Api_Hash,bot_token=Bot_Token)
  return bot,Bot_Identifier

def is_male(Img):
    image = cv2.imread(Img)
    face, confidence = cv.detect_face(image)
    classes = ['man','woman']
    for idx, f in enumerate(face):     
      (startX, startY) = f[0], f[1]
      (endX, endY) = f[2], f[3]
      cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
      face_crop = np.copy(image[startY:endY,startX:endX])
      if face_crop is not None and face_crop.size > 0:
          face_crop = cv2.resize(face_crop, (96,96))
          face_crop = face_crop.astype("float") / 255.0
          face_crop = img_to_array(face_crop)
          face_crop = np.expand_dims(face_crop, axis=0)
          conf = gender_model.predict(face_crop)[0]
          idx = np.argmax(conf)
          label = classes[idx]
          if label == 'man':
              return True
          else :
            return False
      else :
            return False

async def Blur_Female(file_path):
  mainDir = '/'.join(file_path.split('/')[:-1]) + '/'
  P_Name = mainDir + file_path.split('/')[-1].split('.')[0]
  Ex = file_path.split('.')[-1]
  Res_File = f"{P_Name}_Blurred.{Ex}"
  Aud = await Mp3_Conv(file_path)
  file_path = await Media_Compress(file_path)
  cap = cv2.VideoCapture(file_path)
  if not cap.isOpened():
    raise ValueError("Error opening video file")
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(Res_File, fourcc, fps, (width, height))
  while(True):
    ret, frame = cap.read()
    if ret:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      (humans, _) = detect_model.detectMultiScale(frame, winStride=(4, 4),padding=(0, 0), scale=1.05)
      for (x,y,w,h) in humans:
         cv2.imwrite("detected_object.jpg", frame[y:y+h, x:x+w])
         if not await is_male('detected_object.jpg'):
            frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], (51, 51))
      out.write(frame)
    else:
        break 
  cap.release()
  out.release()
  cv2.destroyAllWindows()
  Res_File = await Vid_Mk(Res_File,Aud)
  Res_File = await Media_Compress(Res_File)
  return Res_File

bot,Bot_Identifier = Pyrogram_Client(Bot_Token)
Dl_Dir = f'./Blur_{Bot_Identifier}/'


@bot.on_message(filters.command('start') & filters.private)
async def command1(bot,message):
   await message.reply('لبقية البوتات \n\n @sunnaybots')

@bot.on_message(filters.private & filters.incoming & ( filters.video))
async def _telegram_file(client, message):
  if message.video :
   Reply = await message.reply('جار العمل ...')
   Vid_Path = await message.download(file_name=Dl_Dir)
   Blurred_Vid = await Blur_Female(Vid_Path)
   await message.reply_video(Blurred_Vid)
   await Reply.edit_text('تمت ')
   await Check_Dir(Dl_Dir)


def main():
    if not os.path.exists(Dl_Dir): os.makedirs(Dl_Dir)
    try:
        bot.start()
        print("✅ Blur Bot is ONLINE!")
        idle()
    finally:
        if bot.is_connected:
            bot.stop()

main()

