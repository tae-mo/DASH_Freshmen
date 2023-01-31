
def make_patches(img, p):
    patches = np.array([])
    _, h, w = img.shape
    
    for y in range(0, h, p):
        for x in range(0, w, p):
            if (h-y) < p or (w-x) < p:
                break
            
            tiles = img[:, y:y+p, x:x+p]
            
            if patches.size == 0:
                patches = tiles.reshape(1,-1)
                
            else:
                patches = np.vstack([patches, tiles.reshape(1, -1)])
    
    return patches