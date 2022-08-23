from analise_videos import *
from datetime import datetime


def main():
    for i in glob.glob("Files_Analise\*"):
              
        print(f"########## - {i} - #############")
        try:

            df = analise_video(i)
            nome = i.replace("Files_Analise\\","").replace("\\","-").replace(".","_")
            df.to_pickle(f"Dados\\{nome}-{str(datetime.now()).replace(':','').replace('.','')}.pkl")
        except:
            pass


if __name__ == "__main__":
    main()



