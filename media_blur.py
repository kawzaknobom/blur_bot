import nest_asyncio,os
nest_asyncio.apply()

#########################################################

Bot_Token = os.getenv('TOKEN')

########################################################

from pyrogram import Client, filters,enums,StopTransmission,idle
from pyrogram.types import Message
from ultralytics import YOLO
from deepface import DeepFace
import cv2,os,shutil,time

model = YOLO('yolov8n.pt') 


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

async def get_gender(frame):
    Women_Faces = []
    analysis = DeepFace.analyze(frame, actions=['gender'], detector_backend='retinaface')
    for face in analysis :
       region = face['region']
       x, y, w, h = region['x'], region['y'], region['w'], region['h']
       if face['dominant_gender'] == 'Woman':
        Women_Faces.append((x,y,w,h))
    return Women_Faces

async def get_persons(frame):
   last_known_people = []
   results = model.track(frame, classes=0, persist=True)
   for r in results:
         if r.boxes.id is not None:
            for box, track_id in zip(r.boxes.xyxy, r.boxes.id):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                last_known_people.append((x1, y1, x2, y2))
   return last_known_people

async def is_body(facebbox,bodybboxlist):
            fx1, fy1, fx2, fy2 = facebbox
            for bodybbox in bodybboxlist :
               bx1, by1, bx2, by2 = bodybbox
               if (fx1 >= bx1 and fx2 <= bx2 and fy1 >= fy1 and fy2 <= by2) :
                  return bodybbox
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
  last_update_time = 0
  start_point = False 
  ret_num = 0
  UPDATE_INTERVAL = 5 # seconds
  while(True):
    ret, frame = cap.read()
    if ret:
      ret_num += 1
      if (ret_num%(int(fps)*5) == 0) or start_point == False :
        last_known_people = await get_persons(frame)
        Women_faces = await get_gender(frame)
        if len(Women_faces) == 0 :
          start_point = False      
        else :
          start_point = True    
      
      if len(Women_faces) != 0 :
        for face in Women_faces :
            body = await is_body(face,last_known_people)
            if body :
              x1, y1, x2, y2 = body
              frame[y1:y2,x1:x2] = cv2.blur(frame[y1:y2, x1:x2], (51, 51))

      out.write(frame)
    else:
        break 
  cap.release()
  out.release()
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

