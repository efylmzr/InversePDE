import mitsuba as mi 
import drjit as dr
import sys
import numpy as np

class Slice():
    def __init__(self, axis  : str = "z", offset : float = 0, scale : float = 1):

        transform = mi.ScalarTransform4f()
        self.offset = offset

        if axis == "x":
            transform = transform.rotate(mi.ScalarPoint3d(1,0,0), 90).rotate(mi.ScalarPoint3d(0,1,0), 90)
        elif axis == "y":
            transform = transform.rotate(mi.ScalarPoint3d(0,0,1), -90).rotate(mi.ScalarPoint3d(0,1,0), -90)
        elif axis == "z":
            pass
        else:
            raise Exception("There is no such axis.")
        
        transform = transform.translate(mi.ScalarPoint3d(0, 0, offset)).scale(scale)
        
        self.rectangle = mi.load_dict({
            'type': 'rectangle',
            'material': 
            {
                'type': 'diffuse'
            },
            'to_world' : transform
            })

        self.transform = mi.Transform4f(transform)

    def create_slice_points(self, resolution : list[int], spp : int, seed : int = 64, centered = False) -> mi.Point2f:
        # Generate the first points
 
        x, y = dr.meshgrid(dr.arange(mi.Float, resolution[0]), 
                        dr.arange(mi.Float, resolution[1]), indexing='xy')
        x = dr.repeat(x, spp)
        y = dr.repeat(y, spp)
        if not centered:
            npoints = resolution[0] * resolution[1] * spp
            np.random.seed(seed)
            init_state = np.random.randint(sys.maxsize, size = npoints)
            init_seq = np.random.randint(sys.maxsize, size = npoints)
            sampler = mi.PCG32(npoints, initstate = init_state, initseq = init_seq)
            film_points =  mi.Point2f(x,y) + mi.Point2f(sampler.next_float32(), sampler.next_float32())
        else:
            film_points =  mi.Point2f(x,y) + mi.Point2f(0.5, 0.5)
        # The bounding box is defined as (bottom-left,up-right)
        #points = (mi.Point2f(bbox[0][0], bbox[1][1]) +  
        #        film_points / mi.Point2f(resolution) *  
        #        (mi.Point2f(bbox[1][0], bbox[0][1]) - 
        #        mi.Point2f(bbox[0][0], bbox[1][1])))
        film_points /= mi.Point2f(resolution[0], resolution[1]) 
        film_points = mi.Point2f(film_points[0], 1-film_points[1])
        film_points = film_points * 2 -1
        points = self.transform @ mi.Point3f(film_points[0], film_points[1], 0)
        return points, film_points

    
        
    