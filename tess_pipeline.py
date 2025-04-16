import os
import shutil
import random

# Hardcoded paths instead of config import
TRAINING_FILES_PATH = 'features/'  # Make sure features/Actor_25 and features/Actor_26 exist
TESS_ORIGINAL_FOLDER_PATH = 'tess/'  # <--- Update this with your actual path


class TESSPipeline:

    @staticmethod
    def create_tess_folders(path):
        """
        We are filling folders Actor_25 if YAF and Actor_26 if OAF.
        The files will be copied and renamed and not simply moved (to avoid messing up
        things during the development of the pipeline).
        Actor_25 and Actor_26 folders must be created before launching this script.
        Example filename: 03-01-07-02-02-01-01.wav
        """
        label_conversion = {'01': 'neutral',
                            '03': 'happy',
                            '04': 'sad',
                            '05': 'angry',
                            '06': 'fear',
                            '07': 'disgust',
                            '08': 'ps'}  # 'ps' stands for pleasant surprise

        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if not filename.endswith('.wav'):
                    continue

                if filename.startswith('OAF'):
                    destination_path = os.path.join(TRAINING_FILES_PATH, 'Actor_26')
                elif filename.startswith('YAF'):
                    destination_path = os.path.join(TRAINING_FILES_PATH, 'Actor_25')
                else:
                    continue

                os.makedirs(destination_path, exist_ok=True)
                old_file_path = os.path.join(os.path.abspath(subdir), filename)
                base, extension = os.path.splitext(filename)

                for key, value in label_conversion.items():
                    if base.endswith(value):
                        random_list = random.sample(range(10, 99), 7)
                        file_name = '-'.join([str(i) for i in random_list])
                        file_name_with_correct_emotion = file_name[:6] + key + file_name[8:] + extension
                        new_file_path = os.path.join(destination_path, file_name_with_correct_emotion)
                        shutil.copy(old_file_path, new_file_path)
                        break  # once matched, no need to check other labels


if __name__ == '__main__':
    TESSPipeline.create_tess_folders(TESS_ORIGINAL_FOLDER_PATH)
