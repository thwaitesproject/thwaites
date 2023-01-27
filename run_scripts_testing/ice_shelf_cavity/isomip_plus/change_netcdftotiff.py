import rasterio
from rasterio.transform import Affine


transform = Affine.translation(320000,80000) * Affine.scale(1000, -1000)

with rasterio.open(f'netcdf:Ocean1_input_geom_v1.01.nc:lowerSurface', 'r') as data:
    img = data.read()
    print(img)

with rasterio.open('ocean1_lowersurface.tiff', 'w', 
        driver='GTiff', 
        height=data.shape[0], 
        width=data.shape[1], 
        count=data.count,
        dtype='float64', 
        crs=data.crs, 
        transform=transform) as dst:
    dst.write(img)

with rasterio.open('ocean1_lowersurface.tiff', 'r') as data2:
    img = data2.read()
    print(img)
