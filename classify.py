import re

# Erkennt Zahlen mit Komma auch solche wie 200.000.000
number = re.compile("(-)?\d+((,|.)\d)?")
trunc = re.compile("(\w|[äöüß])+\-")
# https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
link = re.compile(
    "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})")

ovv3 = re.compile("^.{1,3}")
ovv3_start = re.compile("^\^.{1,3}")
ovv3_low = re.compile("^([a-z]|[ßäöü]){1,3}")
ovv3_capital = re.compile("^([A-Z]|[ÄÖÜ]){1}([a-z]|[ßäöü]){1,2}")

ovv6 = re.compile("^.{4,6}")
ovv6_start = re.compile("^\^.{4,6}")
ovv6_low = re.compile("^([a-z]|[ßäöü]){4,6}")
ovv6_capital = re.compile("^([A-Z]|[ÄÖÜ]){1}([a-z]|[ßäöü]|\-([A-Z]|[ÄÖÜ])){3,5}")

ovv9 = re.compile("^.{7,9}")
ovv9_start = re.compile("^\^.{7,9}")
ovv9_low = re.compile("^([a-z]|[ßäöü]){7,9}")
ovv9_capital = re.compile("^([A-Z]|[ÄÖÜ]){1}([a-z]|[ßäöü]|\-([A-Z]|[ÄÖÜ])){6,8}")

ovv_huge = re.compile("^.{8,}")
ovv_huge_start = re.compile("^\^.{9,}")
ovv_huge_low = re.compile("^([a-z]|[ßäöü]){9,}")
ovv_huge_capital = re.compile("^([A-Z]|[ÄÖÜ]){1}([a-z]|[ßäöü]|\-([A-Z]|[ÄÖÜ])){8,}")

classifyers = [
    "OVV3-CAPITAL", "OVV3-LOW", "OVV3-START", "OVV3",
    "OVV6-CAPITAL", "OVV6-LOW", "OVV6-START", "OVV6",
    "OVV9-CAPITAL", "OVV9-LOW", "OVV9-START", "OVV9",
    "OVV-HUGE-CAPITAL", "OVV-HUGE-LOW", "OVV-HUGE-START", "OVV-HUGE"
]

blocking_list = [
    "$(", "ART", "$.", "APPRART", "$,", "PTKNEG", "KOKOM", "VAPP", "VMINF", "PRELAT", "VMPP"
]


def classify_word(word, total):
    if number.fullmatch(word) is not None:
        return "NUMBER"
    elif trunc.fullmatch(word) is not None:
        return "TRUNC"
    elif link.fullmatch(word) is not None:
        return "LINK"

    if total > 1:
        return None

    if ovv3_capital.fullmatch(word) is not None:
        return "OVV3-CAPITAL"
    elif ovv3_low.fullmatch(word) is not None:
        return "OVV3-LOW"
    elif ovv3_start.fullmatch(word) is not None:
        return "OVV3-START"
    elif ovv3.fullmatch(word) is not None:
        return "OVV3"

    elif ovv6_capital.fullmatch(word) is not None:
        return "OVV6-CAPITAL"
    elif ovv6_low.fullmatch(word) is not None:
        return "OVV6-LOW"
    elif ovv6_start.fullmatch(word) is not None:
        return "OVV6-START"
    elif ovv6.fullmatch(word) is not None:
        return "OVV3"

    elif ovv9_capital.fullmatch(word) is not None:
        return "OVV9-CAPITAL"
    elif ovv9_low.fullmatch(word) is not None:
        return "OVV9-LOW"
    elif ovv9_start.fullmatch(word) is not None:
        return "OVV9-START"
    elif ovv9.fullmatch(word) is not None:
        return "OVV3"

    elif ovv_huge_capital.fullmatch(word) is not None:
        return "OVV-HUGE-CAPITAL"
    elif ovv_huge_low.fullmatch(word) is not None:
        return "OVV-HUGE-LOW"
    elif ovv_huge_start.fullmatch(word) is not None:
        return "OVV-HUGE-START"
    elif ovv_huge.fullmatch(word) is not None:
        return "OVV3"

    return None
