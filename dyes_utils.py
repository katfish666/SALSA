
def rgb2hex(rgb):
    rgb = [float(x) for x in rgb]
    rgb = [255*x for x in rgb]
    rgb = [int(x) for x in rgb]
    r,g,b = rgb[0], rgb[1], rgb[2]
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

# https://stackoverflow.com/questions/8915113/sort-hex-colors-to-match-rainbow
import colorsys
def get_hsv(hexrgb):
    hexrgb = hexrgb.lstrip("#") 
    r, g, b = (int(hexrgb[i:i+2], 16) / 255.0 for i in range(0,5,2))
    return colorsys.rgb_to_hsv(r, g, b)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# https://stackoverflow.com/questions/40581878/gradient-with-spectral-lines
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def lin_inter(value, left=0., right=1., increase=True):
    """
    Returns the fractional position of ``value`` between ``left`` and
    ``right``, increasing from 0 if ``value==left`` to 1 if ``value==right``,
    or decreasing from 1 to zero if not ``increase``.
    """
    if increase:
        return (value - left) / (right - left)
    else:
        return (right - value) / (right - left)
    
def wav2RGB(Wavelength, upto255=False, Gamma=1.0):
    
    if isinstance(Wavelength, str):
        Wavelength = float(Wavelength)
    """
    Converts an wavelength to an RGB list, with fractional values between
    0 and 1 if not ``upto255``, or int values between 0 and 255 if ``upto255``
    """
    # Check the interval the color is in
    if 380 <= Wavelength < 440:
        # red goes from 1 to 0:
        Red = lin_inter(Wavelength, 380., 440., False)
        Green = 0.0
        Blue = 1.0
    elif 440 <= Wavelength < 490:
        Red = 0.0
        # green goes from 0 to 1:
        Green = lin_inter(Wavelength, 440., 490., True)
        Blue = 1.0
    elif 490 <= Wavelength < 510:
        Red = 0.0
        Green = 1.0
        Blue = lin_inter(Wavelength, 490., 510., False)
    elif 510 <= Wavelength < 580:
        Red = lin_inter(Wavelength, 510., 580., True)
        Green = 1.0
        Blue = 0.0
    elif 580 <= Wavelength < 645:
        Red = 1.0
        Green = lin_inter(Wavelength, 580., 645., False)
        Blue = 0.0
    elif 645 <= Wavelength <= 780:
        Red = 1.0
        Green = 0.0
        Blue = 0.0
    else: # Wavelength < 380 or Wavelength > 780
        Red = 0.0
        Green = 0.0
        Blue = 0.0
    # Let the intensity fall off near the vision limits
    if 380 <= Wavelength < 420:
        factor = 0.3 + 0.7*lin_inter(Wavelength, 380., 420., True)
    elif 420 <= Wavelength < 700:
        factor = 1.0
    elif 700 <= Wavelength <= 780:
        factor = 0.3 + 0.7*lin_inter(Wavelength, 700., 780., False)
    else:
        factor = 0.0
    # Adjust color intensity
    if upto255:
        def Adjust(Color, Factor):
            return int(round(255. * (Color * Factor)**Gamma))
    else:
        def Adjust(Color, Factor):
            return (Color * Factor)**Gamma
    R = Adjust(Red, factor)
    G = Adjust(Green, factor)
    B = Adjust(Blue, factor)
    #return color
    return [R, G, B]