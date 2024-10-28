
import drjit as dr
import numpy as np
from PDE2D.Coefficient import *
from PDE2D.BoundaryShape import *
from PDE2D.utils import *
from PDE2D import ArrayXb, ArrayXu
from mitsuba import Point2i
import scipy

class CircleWithElectrodes(CircleShape):
    def __init__(self, origin = [0.0, 0.0], radius = 1.0, name = "electrodeCircle", epsilon = 1e-5,
                 num_electrodes = 16, is_delta = False, electrode_length = 0.01, 
                 injection_confs = [[0,1]], injected_current = 1.0, electrode_potentials = None,
                 offset_angle = 0.0, centered = False, fileset = None, injection_set = None, delete_injection = True):
        self.name = name
        super().__init__(origin, radius, dirichlet_map = np.array([False]), epsilon = epsilon, name = self.name)
        if fileset is not None:
            mat = scipy.io.loadmat(fileset)
            range_exp = self.get_injection_range_file_all(fileset=fileset, injection_sets=injection_set)
            self.measured_current = True
            self.voltages_first = mat["Uel"].T[range_exp]
            self.num_confs = self.voltages_first.shape[0]
            self.num_electrodes = self.voltages_first.shape[1]
            self.voltages_first = np.hstack([np.zeros([self.num_confs,1]), self.voltages_first])
            self.voltages_first = self.voltages_first.cumsum(axis = 1)[:, :16]
            self.currents = mat["CurrentPattern"].T[range_exp]
            nonzeros = np.nonzero(self.currents)
            if delete_injection:
                self.voltages = self.voltages_first
                self.voltages[self.currents!=0] = 0
                self.voltages = self.voltages - (np.sum(self.voltages, axis = 1))[:,np.newaxis] / (num_electrodes - 2)
                self.voltages[self.currents!=0] = 0
            else:
                self.voltages = self.voltages_first
                self.voltages = self.voltages - (np.mean(self.voltages, axis = 1))[:,np.newaxis]
            #self.voltages = self.voltages.astype(np.float32)
            self.voltages_std = np.zeros(num_electrodes)
            positive_inj = nonzeros[1][np.nonzero((self.currents[nonzeros] > 0).astype(np.int16) )]
            negative_inj = nonzeros[1][np.nonzero((self.currents[nonzeros] < 0).astype(np.int16) )]
            current_confs = np.vstack([positive_inj, negative_inj]).T
            self.electrode_length = 0.025
            self.injected_current = np.abs(self.currents[nonzeros][0])
            self.injections = current_confs
            self.injection_confs = Point2i(current_confs.T)
        else:
            self.num_electrodes = num_electrodes
            
            if injection_set is None:
                if injection_confs is not None:
                    self.injections = injection_confs 
                else:
                    raise Exception("Either specify an injection set or injection configuration.")
            else:
                self.injections = self.create_injection_set_all(injection_set, num_electrodes)
            self.num_confs = len(self.injections)
            self.injection_confs = Point2i(np.array(self.injections).T) # The first one is injected, the second one is received.
            
            self.injected_current = injected_current
            self.voltages = electrode_potentials
            self.electrode_length = electrode_length 
        
        self.is_delta = is_delta
        self.has_delta = is_delta
        self.NEE = NEE.Special
        self.has_continuous_neumann = not is_delta
        self.el_diff_angle = 2 * dr.pi / self.num_electrodes
        self.el_center_angles = correct_angle(offset_angle +  dr.arange(Float, self.num_electrodes) * self.el_diff_angle)
        
        self.normal_ders = {}
        if not is_delta:
            self.el_angle = self.electrode_length / self.radius
            if not centered:
                self.el_center_angles += self.el_angle/2
                self.el_center_angles = correct_angle(self.el_center_angles)
            el_ending1 = correct_angle(self.el_center_angles - self.el_angle/2)
            el_ending2 = correct_angle(self.el_center_angles + self.el_angle/2) 
            self.el_endings = Point2f(el_ending1, el_ending2)
            dr.make_opaque(self.el_endings)
            
        dr.make_opaque(self.el_center_angles)
        self.num_conf_n = self.num_confs
        

    def create_injection_set_all(self, injection_sets, num_electrodes):
        sets = injection_sets.split("-")
        final_set = []
        for set in sets:
            final_set.extend(self.create_injection_set(set, num_electrodes))
        return final_set

    def create_injection_set(self, injection_set, num_electrodes):
        if injection_set == "adjacent":
            set = [[i, (i + 1) % num_electrodes] for i in range(num_electrodes)]
        elif injection_set[:4] == "skip":
            try:
                skip =  int(injection_set[4:])
            except:
                print("You need to specify a number after skip.")
            set = [[i, (i + skip + 1) % num_electrodes] for i in range(num_electrodes)]
        elif injection_set[:7] == "against":
            try:
                against =  int(injection_set[7:])
            except:
                print("You need to specify a number after against.")
            set = [[against, (against + i) % num_electrodes] for i in range(num_electrodes - 1)]
        else:
            raise Exception("There is no such injection set.")
        return set

    def get_injection_range_file_all(self, fileset, injection_sets):
        sets = injection_sets.split("-")
        range_all = []
        for set in sets:
            range_all.extend(self.get_injection_range_file(fileset, set))
        return range_all

    def get_injection_range_file(self, fileset, injection_set : str):
        if injection_set == "adjacent":
            range_exp = [i for i in range(0, 16)]
        elif injection_set == "skip1":
            range_exp = [i for i in range(16, 32)]
        elif injection_set == "skip2":
            range_exp = [i for i in range(32, 48)]
        elif injection_set == "skip3":
            range_exp = [i for i in range(48, 64)]
        elif injection_set == "against1":
            range_exp = [i for i in range(64, 79)]
        elif injection_set == "all":
            range_exp = [i for i in range(0, 79)]
        else:
            raise Exception("There is no such injection!")
        return range_exp
    
    def get_injection_confs(self, allsets : str, vis_set, num_electrodes : int):
        sets = allsets.split("-")
        range_all = []
        begin = 0
        end = 0
        found = False
        for set in sets:
            if set == vis_set:
                set = self.create_injection_set(set, num_electrodes)
                found = True
                begin = len(range_all)
                end = begin + len(set)
            else:
                range_all.extend(self.create_injection_set(set, num_electrodes))
        if not found:
            raise Exception("Such set does not exist")
        else:
            return [dr.opaque(UInt32, i, shape = (1)) for i in range(begin, end)]
    
    @dr.syntax        
    def sampleNEE(self, bi : BoundaryInfo, sample : Float, conf_number : UInt32) -> tuple[Float, Float, Float, Point2f]:
        d, n, pdf_r, sampled = (Float(0), Float(0), Float(0), Point2f(0))
        if dr.hint(self.has_continuous_neumann, mode = 'scalar'):
            d, n, pdf_r, sampled  = (Float(0), Float(0), Float(0) , Point2f(0))
            if sample < 0.5:
                sample *= 2
                d, n, pdf_r, sampled = self.sample_electrode(bi, sample, conf_number, injected = True)
            else:
                sample = 2 * (sample - 0.5) 
                d, n, pdf_r, sampled = self.sample_electrode(bi, sample, conf_number, injected = False)
        return d, n, pdf_r/2, sampled
        
    def get_point_neumann(self, bi : BoundaryInfo, conf_number : UInt32) -> tuple[list[Float], list[Float], list[Float], list[Point2f]]:
        if self.has_delta:
            d1, n1, pdf1_r, sampled1 = self.sample_electrode(bi, Float(0), conf_number, injected = True)
            d2, n2, pdf2_r, sampled2 = self.sample_electrode(bi, Float(0), conf_number, injected = False)
        return [d1, d2], [n1, n2], [pdf1_r, pdf2_r], [sampled1, sampled2]

    def sample_electrode(self, bi : BoundaryInfo, sample : Float, conf_number : UInt32 , injected = True):
        sign = 1 if injected else -1
        electrode_num = 0 if injected else 1
        # This function assumes there is only 2 electrode injection
        current_conf = dr.gather(Point2i, self.injection_confs, conf_number)
        diff1 = bi.x1 - self.origin
        diff2 = bi.x2 - self.origin
        star_angle1 = correct_angle(dr.atan2(diff1[0], diff1[1]))
        star_angle2 = correct_angle(dr.atan2(diff2[0], diff2[1]))
        if self.is_delta:
            s = dr.gather(Float, self.el_center_angles, current_conf[electrode_num])
            valid =  self.inside_range(star_angle1, star_angle2, s)
            neumann = dr.select(valid & bi.is_star, self.injected_current, 0) * sign
            sampled_point = self.origin + self.radius * Point2f(dr.sin(s), dr.cos(s))
            distance = dr.norm(sampled_point - bi.origin)
            pdf_r = 2 * dr.pi * distance # actual pdf is "1", we multiply everything by 2 pi r, (cancels out Green's function computation)
        else: # either one of the electrode ends are inside the star or the whole electrode covers the star.
            el_end1 = dr.gather(Float, self.el_endings[0], current_conf[electrode_num])
            el_end2 = dr.gather(Float, self.el_endings[1], current_conf[electrode_num])
            el_end1_inside = self.inside_range(star_angle1, star_angle2, el_end1)
            el_end2_inside = self.inside_range(star_angle1, star_angle2, el_end2)
            el_active = bi.is_star & (el_end1_inside | el_end2_inside | self.inside_range(el_end1, el_end2, star_angle1))
            sample_range1 = dr.select(el_end1_inside, el_end1, star_angle1)
            sample_range2 = dr.select(el_end2_inside, el_end2, star_angle2)
            current_flux = self.injected_current / self.electrode_length
            neumann = dr.select(el_active, current_flux, 0) * sign
            # We are going to sample an angle from the star center! So we find the range of angles first
            sample_p_range1 = self.origin + self.radius * Point2f(dr.sin(sample_range1), dr.cos(sample_range1))
            sample_p_range2 = self.origin + self.radius * Point2f(dr.sin(sample_range2), dr.cos(sample_range2))
            range_vec1 = sample_p_range1 - bi.origin
            range_vec2 = sample_p_range2 - bi.origin
            angle1 = correct_angle(dr.atan2(range_vec1[0], range_vec1[1]))
            angle2 = correct_angle(dr.atan2(range_vec2[0], range_vec2[1]))
            bi.update_angles(angle1, angle2)
            # We also need to change on_boundary value. We sample as if we are on the boundary only 
            # if the star origin is on the boundary and inside electrode!
            angle_n = correct_angle(dr.atan2(bi.bn[0], bi.bn[1]))
            star_origin_angle = correct_angle(angle_n + dr.pi) 
            on_boundary_electrode = bi.on_boundary & self.inside_range(el_end1, el_end2, star_origin_angle)
            direction, pdf = bi.sample_neumann(sample, on_boundary_electrode)
            # distance, sampled_point, normals = self.ray_intersect(bi.origin, direction, bi.on_boundary)
            ri = self.ray_intersect(bi, direction)
            pdf_r = pdf * dr.abs(dr.dot(direction, ri.normal)) * 2 * dr.pi # pdf with respect to area and also multiplied with 2 pi r
            distance = ri.t
            sampled_point = ri.intersected
        return distance, neumann, pdf_r, sampled_point
            
    def inside_range(self, angle1, angle2, angle):
        electrode_start = angle1 > angle2
        normal_case = (angle1 < angle) & (angle2 > angle)
        start_case = (angle1 < angle) | (angle2 > angle)
        return  dr.select(electrode_start, start_case, normal_case)

    def create_neumann_function(self, conf_numbers : list[UInt32]):
        if self.is_delta:
            raise NotImplementedError
        confs = []
        for conf_number in conf_numbers: 
            params = {}
            params["conf"] = conf_number
            def neumann_val(point, params):
                injections = dr.gather(Point2i, self.injection_confs, params["conf"])
                el1_ending1 = dr.gather(Float, self.el_endings[0], injections[0])
                el1_ending2 = dr.gather(Float, self.el_endings[1], injections[0])
                el2_ending1 = dr.gather(Float, self.el_endings[0], injections[1])
                el2_ending2 = dr.gather(Float, self.el_endings[1], injections[1])
                diff = point - self.origin
                angle_point = correct_angle(dr.atan2(diff[0], diff[1]))
                inside_el1 = self.inside_range(el1_ending1, el1_ending2, angle_point)
                inside_el2 = self.inside_range(el2_ending1, el2_ending2, angle_point)
                neumann_val = self.injected_current / self.electrode_length
                result = dr.select(inside_el1, neumann_val, 0)
                result = dr.select(inside_el2, -neumann_val, result)
                return result
            neumann_coeff = FunctionCoefficient(f"neumann-{conf_number}", params, neumann_val)
            confs.append(neumann_coeff)
        return confs
    
    def create_electrode_points(self, spe, conf_numbers : list[UInt32], delete_injection : bool = True):
        
        angles = Float(self.el_center_angles)
        points = self.origin + self.radius * Point2f(dr.sin(angles), dr.cos(angles))
        dr.make_opaque(points)
        points = dr.repeat(points, spe)

        electrode_nums = dr.zeros(ArrayXu, shape = (len(conf_numbers), self.num_electrodes))
        active_confs = dr.zeros(ArrayXb, shape=(len(conf_numbers), self.num_electrodes))
        for i, conf_number in enumerate(conf_numbers):
            electrode_num = np.arange(self.num_electrodes)
            active_conf = np.zeros(self.num_electrodes, dtype = bool)
            if delete_injection:
                current_conf = dr.gather(Point2i, self.injection_confs, conf_number)
                electrode_num = np.delete(electrode_num, current_conf.numpy())
                active_conf[electrode_num] = True
            electrode_nums[i] = UInt32(electrode_num)
            active_confs[i] = Bool(active_conf)
        
        dr.make_opaque(active_confs)
        active_confs = dr.repeat(active_confs, spe)
        dr.make_opaque(electrode_nums)
        return points, active_confs, electrode_nums
    
    def sketch(self, ax, bbox, resolution, colors = ["orange", "green"], lw = 3, e_size = None):
        origin_s = point2sketch(self.origin, bbox, resolution)
        origin_s = np.array([origin_s[0][0], origin_s[1][0]])
        radius_x, radius_y, radius = dist2sketch(self.radius, bbox, resolution)
        radius_x = radius_x.numpy()[0]
        radius_y = radius_y.numpy()[0]
        sphere = patches.Ellipse(origin_s, radius_x * 2, radius_y * 2, linewidth= lw,
                                fill = False, color = colors[0], label = self.name)
        ax.add_patch(sphere)

        origin_s = point2sketch(self.origin, bbox, resolution)
        origin_s = np.array([origin_s[0][0] - 0.5, origin_s[1][0] - 0.5])
        radius_x, radius_y, radius = dist2sketch(self.radius, bbox, resolution)
        radius_x = radius_x.numpy()[0]
        radius_y = radius_y.numpy()[0]
        if self.is_delta:
            el_points = point2sketch(self.origin + self.radius * Point2f(dr.sin(self.el_center_angles), dr.cos(self.el_center_angles)),
                                     bbox, resolution).numpy().squeeze()
            if e_size is None:
                ax.scatter(el_points[0,:] -0.5, el_points[1,:]-0.5, color = colors[1])
            else:
                ax.scatter(el_points[0,:] -0.5, el_points[1,:]-0.5, color = colors[1], s = e_size)
        else:
            angles1 = self.el_endings.numpy()[0, :] * 180 / np.pi
            angles2 = self.el_endings.numpy()[1, :] * 180 / np.pi
            for angle1, angle2 in zip(angles1, angles2):
                neumann_arc = patches.Arc(origin_s,  2 * radius_x, 2 * radius_y, angle = -90, theta1=angle1, 
                                          theta2=angle2, linewidth = lw, color = colors[1])
                ax.add_patch(neumann_arc)  

    def sketch_electrode_input(self, ax, bbox, resolution, conf_number = UInt32(0), color = "red"):
        current_conf = dr.gather(Point2i, self.injection_confs, conf_number)
        angle1 = dr.gather(Float, self.el_center_angles, current_conf[0])
        angle2 = dr.gather(Float, self.el_center_angles, current_conf[1])
        point1_start = point2sketch(self.origin + self.radius * 1.1 * Point2f(dr.sin(angle1), dr.cos(angle1)), bbox, resolution).numpy().squeeze()
        point1_diff = dir2sketch(-0.08 * self.radius * Point2f(dr.sin(angle1), dr.cos(angle1)), bbox, resolution).numpy().squeeze()
        point2_start = point2sketch(self.origin + self.radius * 1.02 * Point2f(dr.sin(angle2), dr.cos(angle2)), bbox, resolution).numpy().squeeze()
        point2_diff = dir2sketch(self.radius * 0.08 * Point2f(dr.sin(angle2), dr.cos(angle2)), bbox, resolution).numpy().squeeze()
        arrow_1 = patches.FancyArrow(point1_start[0] - 0.5, point1_start[1] - 0.5, point1_diff[0], point1_diff[1], 
                                    width=2 / 512 * resolution[0], length_includes_head=True, color = color)
        arrow_2 = patches.FancyArrow(point2_start[0] - 0.5, point2_start[1] - 0.5, point2_diff[0], point2_diff[1], 
                                    width=2 / 512 * resolution[0], length_includes_head=True, color = color)
        ax.add_patch(arrow_1)
        ax.add_patch(arrow_2)
        
    