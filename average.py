def compute_average(imlist):
    """ Compute the average of a list of images. """

    # open first image and make into array of type float
    averageim = array(Image.open(imlist[0]), 'f')

    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print imname + '...skipped'
    aveageim /= len(imlist)

    # return average as uint8
    return array(averageim, 'uint8')
