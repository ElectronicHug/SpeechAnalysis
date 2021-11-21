import pandas as pd
import glob

def create_dataframe_from_markup_stream(folder_path):

    markups_list = []

    markup_filenames = glob.glob(folder_path+'/*/*.txt')
    for ind_txt_file, path_txt_file in enumerate(markup_filenames):
        try:
            #Some files broken :(
            one_conversation = pd.read_csv(path_txt_file, sep='	', header=None, names=[1,2,3,4])
            one_conversation.columns = ['phrase_start', 'phrase_stop', 'speaker_language_emotion', 'phrase_text']
            one_conversation['speaker'] = one_conversation['speaker_language_emotion'].apply(lambda row: row[0])
            one_conversation=one_conversation[one_conversation['speaker'] != 'b']
            one_conversation['language'] = one_conversation['speaker_language_emotion'].apply(lambda row: row[2])
            one_conversation['emotion'] = one_conversation['speaker_language_emotion'].apply(lambda row: row[4])
            one_conversation['audio_filepath'] = path_txt_file[:-4]+'.wav'

            one_conversation.drop(columns=['speaker_language_emotion'], inplace=True)

            # in makrup с is from eng and ukr keyboard
            one_conversation['speaker'] = one_conversation['speaker'].replace({'с':'customer', 'c':'customer', 's':'sales'})

            one_conversation['language'] = one_conversation['language'].replace({'r':'rus', 'u':'ukr', 's':'surjik'})

            emotion_short_naming_dict = {"n":"neutral",
            "s":"sad",
            "a":"angry",
            "h":"happy",
            "p":"pleasant_surprise",
            "d":"disgust",
            "f":"fear"}

            one_conversation['emotion'] = one_conversation['emotion'].replace(emotion_short_naming_dict)

            markups_list.append(one_conversation)
        except:

            pass
    markups_df = pd.concat(markups_list)

    return markups_df



def create_dataframe_both_communicate_from_markup_stream(folder_path):
    markups_list = []

    markup_filenames = glob.glob(folder_path+'/*/*.txt')
    for ind_txt_file, path_txt_file in enumerate(markup_filenames):
        try:
            #Some files broken :(
            one_conversation = pd.read_csv(path_txt_file, sep='	', header=None, names=[1,2,3,4])
            one_conversation.columns = ['phrase_start', 'phrase_stop', 'speaker_language_emotion', 'phrase_text']

            def detect_both_communicate(speaker_str):
                if speaker_str == 'b':
                    return True
                else:
                    return False

            one_conversation['both_communicate'] = one_conversation['speaker_language_emotion'].apply(detect_both_communicate)

            one_conversation['audio_filepath'] = path_txt_file[:-4]+'.wav'

            one_conversation.drop(columns=['speaker_language_emotion', 'phrase_text'], inplace=True)

            markups_list.append(one_conversation)
        except:

            pass
    markups_df = pd.concat(markups_list)

    return markups_df