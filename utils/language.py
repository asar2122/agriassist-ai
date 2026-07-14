def detect_language(text):

    tamil_start = 0x0B80

    tamil_end = 0x0BFF


    for character in text:

        character_code = ord(character)

        if tamil_start <= character_code <= tamil_end:

            return "ta"


    return "en"