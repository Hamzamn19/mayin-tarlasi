"""50 new candidate features for landmine detection — 6 groups."""
import cv2
import numpy as np

def _get_contour(gray_crop):
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts: return None, binary
    return max(conts, key=cv2.contourArea), binary

def features_morphological(gray_crop):
    try:
        h, w = gray_crop.shape
        cnt, _ = _get_contour(gray_crop)
        if cnt is None: return [0.0]*8
        ca = cv2.contourArea(cnt); perim = cv2.arcLength(cnt, True)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.int32)
        mra = cv2.contourArea(box)
        hull = cv2.convexHull(cnt); ha = cv2.contourArea(hull); hp = cv2.arcLength(hull, True)
        major = minor = 1.0
        if len(cnt) >= 5:
            (_, _), (ax1, ax2), _ = cv2.fitEllipse(cnt)
            major, minor = max(ax1, ax2), min(ax1, ax2)
        defect_depths = [0.0]
        try:
            if len(cnt) > 3 and len(hull) > 3:
                hi = cv2.convexHull(cnt, returnPoints=False)
                defs = cv2.convexityDefects(cnt, hi)
                if defs is not None: defect_depths = [d[0][3]/256.0 for d in defs]
        except: pass
        r = float(ca / max(mra, 1))
        n = float(min(major, minor) / max(major, minor, 1e-6))
        sd = float(hp**2 / max(4*np.pi*ha, 1e-6))
        epsilon = 0.005 * perim
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        angles = []
        for i in range(len(approx)):
            p1 = approx[(i-1)%len(approx)][0]; p2 = approx[i][0]; p3 = approx[(i+1)%len(approx)][0]
            v1, v2 = p1-p2, p3-p2
            dot = float(v1[0]*v2[0]+v1[1]*v2[1])
            n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            angles.append(np.arccos(np.clip(dot/(n1*n2+1e-10), -1, 1)))
        cr = float(np.std(angles)) if angles else 0.0
        cd = float(np.mean(defect_depths)) if defect_depths else 0.0
        cp = float(4*np.pi*ca / max(perim**2, 1e-6))
        el = float(1 - minor/max(major, 1e-6))
        bb = float(len([d for d in defect_depths if d > 0.1]))
        return [r, n, sd, cr, cd, cp, el, bb]
    except: return [0.0]*8

def features_statistical(gray_crop, cnt=None):
    try:
        f = gray_crop.astype(float).flatten()
        mu, s = np.mean(f), np.std(f)
        p25, p50, p75 = np.percentile(f, [25, 50, 75])
        cv = float(s / max(mu, 1e-6))
        iqr = float(p75 - p25)
        rng = float(f.max() - f.min())
        hist = cv2.calcHist([gray_crop], [0], None, [64], [0, 256]).flatten()
        mode = float(np.argmax(hist) * 4)
        mm = float(hist.max() / max(hist.sum(), 1))
        hist_n = hist / max(hist.sum(), 1)
        hv = float(np.var(hist_n))
        bi = float(1 - hv / max(np.var(np.linspace(0,1,64)), 1e-10))
        eh = float(-np.sum(hist_n * np.log(hist_n + 1e-10)))
        snr = float(mu / max(s, 1e-6))
        return [cv, iqr, rng, mode, mm, bi, eh, snr]
    except: return [0.0]*8

def features_texture(gray_crop):
    try:
        h, w = gray_crop.shape
        gq = (gray_crop/8).astype(np.uint8); L=32
        glcm = np.zeros((L,L), dtype=np.float64)
        for i in range(h): np.add.at(glcm, (gq[i,:-1], gq[i,1:]), 1)
        total = glcm.sum()
        if total==0: return [0.0]*8
        glcm /= total; ii,jj = np.meshgrid(np.arange(L),np.arange(L),indexing='ij')
        d = np.abs(ii-jj)
        dis = float(np.sum(glcm * d))
        mp = float(glcm.max())
        mu_g = np.sum(ii*glcm)
        gv = float(np.sum(((ii-mu_g)**2)*glcm))
        sa = float(np.sum((ii+jj)*glcm))
        ks, sigma, lamda, gamma = 21, 3.0, 5.0, 0.5
        energies = []
        for th in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            k = cv2.getGaborKernel((ks,ks), sigma, th, lamda, gamma, psi=0)
            r = cv2.filter2D(gray_crop, cv2.CV_64F, k)
            energies.append(np.sum(r**2))
        ge = float(np.sum(energies)); gm = float(np.mean(energies))
        lbp = np.zeros((h-2,w-2), dtype=np.uint8)
        for i in range(1,h-1):
            for j in range(1,w-1):
                c=gray_crop[i,j]; code=0
                for k,(di,dj) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
                    code |= (gray_crop[i+di,j+dj]>=c) << k
                mc = code
                for r in range(1,8): mc = min(mc, ((code>>r)|(code<<(8-r)))&0xFF)
                lbp[i-1,j-1]=mc
        lr = float(len(np.unique(lbp)))/256.0 if lbp.size>0 else 0.0
        lc = float(np.var(lbp.astype(float))) if lbp.size>0 else 0.0
        return [dis, mp, gv, sa, ge, gm, lr, lc]
    except: return [0.0]*8

def features_frequency(gray_crop):
    try:
        f = gray_crop.astype(float); h,w = f.shape
        hp, wp = (h+1)//2*2, (w+1)//2*2
        fpad = np.zeros((hp,wp),dtype=f.dtype); fpad[:h,:w]=f
        dct = cv2.dct(fpad/255.0); tdct = np.sum(np.abs(dct))+1e-10
        n = max(1, min(hp,wp)//8)
        low = float(np.sum(np.abs(dct[:n,:n]))/tdct)
        high = float(np.sum(np.abs(dct[-n:,-n:]))/tdct)
        def haar(img):
            r,c = img.shape; r-=r%2; c-=c%2; img=img[:r,:c]
            L = (img[:,::2]+img[:,1::2])/2; H = (img[:,::2]-img[:,1::2])/2
            return (L[::2,:]+L[1::2,:])/2, (L[::2,:]-L[1::2,:])/2, (H[::2,:]+H[1::2,:])/2, (H[::2,:]-H[1::2,:])/2
        LL,LH,HL,HH = haar(fpad) if min(h,w)>=4 else (np.array([[0]]),)*4
        te = float(np.sum(LL**2)+np.sum(LH**2)+np.sum(HL**2)+np.sum(HH**2))+1e-10
        wa = float(np.sum(LL**2)/te); wh = float(np.sum(LH**2)/te)
        wv = float(np.sum(HL**2)/te); wd = float(np.sum(HH**2)/te)
        fft = np.fft.fft2(fpad); fshift = np.fft.fftshift(fft); mag = np.abs(fshift)
        cyf,cxf=hp//2,wp//2; maxr=min(cyf,cxf)
        yy,xx=np.ogrid[:hp,:wp]; dist=np.sqrt((yy-cyf)**2+(xx-cxf)**2).astype(int)
        bins_w = min(20, maxr)
        radial = [np.mean(mag[(dist>=r)&(dist<r+maxr//bins_w)]) for r in range(0,maxr-1,maxr//bins_w)] if maxr>0 else [0]
        rp = float(np.argmax(radial)/max(len(radial),1)) if radial else 0.0
        angles = np.arctan2(np.arange(hp)[:,None]-cyf, np.arange(wp)-cxf)
        abins=12; ahist=np.zeros(abins)
        for b in range(abins):
            mask=(angles>=-np.pi+b*2*np.pi/abins)&(angles<-np.pi+(b+1)*2*np.pi/abins)
            ahist[b]=np.mean(mag[mask]) if mask.any() else 0
        an=float(np.std(ahist)/max(np.mean(ahist),1e-10))
        return [low, high, wa, wh, wv, wd, rp, an]
    except: return [0.0]*8

def features_multiscale(gray_crop):
    try:
        f = gray_crop.astype(float)
        log2=cv2.Laplacian(cv2.GaussianBlur(gray_crop,(7,7),2),cv2.CV_64F); ls2=float(np.mean(np.abs(log2)))
        log4=cv2.Laplacian(cv2.GaussianBlur(gray_crop,(13,13),4),cv2.CV_64F); ls4=float(np.mean(np.abs(log4)))
        g1=cv2.GaussianBlur(gray_crop,(5,5),1).astype(float); g2=cv2.GaussianBlur(gray_crop,(13,13),3).astype(float)
        dr=float(np.std(g1-g2))
        gx=cv2.Sobel(f,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(f,cv2.CV_64F,0,1,ksize=3)
        gxx=cv2.Sobel(gx,cv2.CV_64F,1,0,ksize=3); gyy=cv2.Sobel(gy,cv2.CV_64F,0,1,ksize=3)
        gxy=cv2.Sobel(gx,cv2.CV_64F,0,1,ksize=3)
        det=gxx*gyy-gxy**2; tr=gxx+gyy
        hd=float(np.mean(np.abs(det))); ht=float(np.mean(np.abs(tr)))
        sh=float(np.mean(np.abs(gx))); sv=float(np.mean(np.abs(gy)))
        kx=np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=float)
        ky=np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype=float)
        px=cv2.filter2D(f,cv2.CV_64F,kx); py=cv2.filter2D(f,cv2.CV_64F,ky)
        pm=float(np.mean(np.sqrt(px**2+py**2)))
        return [ls2, ls4, dr, hd, ht, sh, sv, pm]
    except: return [0.0]*8

def features_geometric(gray_crop):
    try:
        h,w=gray_crop.shape; f=gray_crop.astype(float)
        cnt,_=_get_contour(gray_crop)
        if cnt is None: return [0.0]*10
        ca=cv2.contourArea(cnt); perim=cv2.arcLength(cnt,True)
        la=float(np.log10(ca+1)); sa=float(np.sqrt(ca))
        par=float(perim/max(ca,1))
        x,y,wb,hb=cv2.boundingRect(cnt); br=float(ca/max(wb*hb,1))
        fd=0.0
        try:
            hp_pts=cv2.convexHull(cnt).squeeze()
            if hp_pts.ndim==2 and len(hp_pts)>=2:
                diffs=hp_pts[:,None,:]-hp_pts[None,:,:]
                fd=float(np.sqrt(np.sum(diffs**2,axis=-1)).max())
        except: pass
        mf=0.0
        if len(cnt)>=5:
            _, (ax1,ax2),_=cv2.fitEllipse(cnt); mf=float(min(ax1,ax2))
        _,binary=_get_contour(gray_crop)
        cy,cx=h//2,w//2; Ys,Xs=np.where(binary>0)
        max_r=float(np.max(np.sqrt((Ys-cy)**2+(Xs-cx)**2))) if len(Ys)>0 else 1
        gc=float(ca/max(np.pi*max_r**2,1))
        inv=cv2.bitwise_not(binary)
        hc_conts,_=cv2.findContours(inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        hc=float(len(hc_conts)-1) if len(hc_conts)>1 else 0.0
        lap=cv2.Laplacian(gray_crop,cv2.CV_64F)
        zc=float(np.log10(np.sum((lap[:-1,:-1]*lap[1:,1:]<0).astype(float))+1))
        at=cv2.adaptiveThreshold(gray_crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        am=float(np.mean(at))/255.0
        return [la, sa, par, br, fd, mf, gc, hc, zc, am]
    except: return [0.0]*10

def compute_all_50(gray_crop):
    m = features_morphological(gray_crop)
    s = features_statistical(gray_crop)
    t = features_texture(gray_crop)
    f = features_frequency(gray_crop)
    ml = features_multiscale(gray_crop)
    g = features_geometric(gray_crop)
    return m + s + t + f + ml + g

FEATURE_NAMES_50 = [
    "rectangularity","narrowness","shape_dispersion","contour_roughness",
    "convexity_defect_depth","compactness","elongation","blobbiness",
    "cv_intensity","iqr_intensity","intensity_range","intensity_mode",
    "intensity_mode_magnitude","bimodality_index","entropy_histogram","signal_to_noise",
    "glcm_dissimilarity","glcm_max_prob","glcm_variance","glcm_sum_avg",
    "gabor_energy","gabor_mean","lbp_ri","lbp_contrast",
    "dct_low_energy","dct_high_energy","wavelet_approx","wavelet_h_detail",
    "wavelet_v_detail","wavelet_d_detail","fft_radial_peak","fft_angular_std",
    "log_sigma2","log_sigma4","dog_ratio","hessian_det",
    "hessian_trace","sobel_horizontal","sobel_vertical","prewitt_mean",
    "log_area","sqrt_area","perimeter_area_ratio","bounding_box_ratio",
    "feret_diameter","min_feret","geometric_compactness","hole_count",
    "log_zero_crossings","adaptive_threshold_mean",
]
