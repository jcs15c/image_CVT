import cvt_lib as cvt
import cvt_combo as cvt2
from scipy import misc
import matplotlib.pyplot as plt


def main():
    #load Image
    imname = "bird"
    
    ########################### Combination CVT ###############################
    c_data, g_data = cvt2.read_image_combo("images/" + imname + ".png")
    
    c_gens = cvt2.plusplus(c_data, 12, 3)
    g_gens = cvt2.plusplus(g_data, 12, 2)

    c_gens, g_gens, sketch_comb = \
            cvt2.cvt_combo(c_data, g_data, c_gens, g_gens, 1e-4, 30, 2, 2.3)
    cvt_comb = cvt2.cvt_render_combo(c_data, g_data, c_gens, g_gens, 2, 2.3)
    
    misc.imsave("combo_images/" + imname + "_combo_CVT.png", cvt_comb)
    misc.imsave("combo_images/" + imname + "_combo_SEG.png", sketch_comb)

    ############################ Ordinary CVT #################################
    data = cvt.read_image("images/" + imname + ".png")
    
    o_gens = cvt.plusplus(data,2)
    
    sketch_ord, generators_new, weights, E = \
                cvt.image_segmentation(data, o_gens, 1e-4, 30, 0)
    cvt_ord = cvt.cvt_render(data, generators_new, weights, 0)

    misc.imsave("combo_images/" + imname + "_ord_CVT.png", cvt_ord)
    misc.imsave("combo_images/" + imname + "_ord_SEG.png", sketch_ord)

    ############################## Plotting ###################################
    plt.figure(1, figsize=(15, 9))
    
    plt.subplot(231)
    plt.title("Original image")
    plt.imshow(data)

    plt.subplot(232)
    plt.title("Ord. CVT w/ 2 Colors")
    plt.imshow(cvt_ord/256)
    
    plt.subplot(233)
    plt.title("Ord. Segmentation")
    plt.imshow(sketch_ord, cmap='gray')
    
    plt.subplot(235)
    plt.title("Combo CVT Colors")
    plt.imshow(cvt_comb/256)

    plt.subplot(236)
    plt.title("Combo Segmentation")
    plt.imshow(sketch_comb, cmap='gray')

    plt.savefig("combo_images/Combo_Example_" + imname + ".png")

    return 0

main()  



