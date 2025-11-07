#
# Created by: Anil N Hirani
# Timestamp: 2009-12-04 04:55:55 -0600 (Fri, 04 Dec 2009)
#

__all__ = ['visualizable_simplicial_mesh','meshqv']

from numpy import (ravel, average, zeros, abs, nonzero, array,
                       sort, lexsort, ones, sign)
from numpy import flipud, mean, loadtxt
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Circle
from pydec import (simplicial_mesh, circumcenter, simplicial_complex,
                       Simplex)
from itertools import combinations

try:
    from mayavi import mlab as mmlab
except:
    print('CAUTION : Mayavi not available ? ' +
          'Solid and surface plotting may not work.\n\n')


class visualizable_simplicial_mesh(simplicial_complex):

    def __init__(self, *args):
        simplicial_complex.__init__(self, *args)
        if (self.embedding_dimension() == 2 and self.complex_dimension() == 2):
            self.mesh_type = 'planar'
            self.edgecolor = (0, 0, 0)
            self.facecolor = (1, 1, 1)
            self.dualedgecolor = (0, 0, 1)
            self.dualfacecolor = (1, 1, 1)
            self.overdualedgecolor = (1, 1, 1)
        elif (self.embedding_dimension() == 3 and self.complex_dimension() == 3):
            self.mesh_type = 'volume'
            self.edgecolor = (0,0,0)
            self.facecolor = None
        elif (self.embedding_dimension() == 3 and self.complex_dimension() == 2):
            self.mesh_type = 'surface'
            self.edgecolor = (0.8, 0.8, 0.8)
            self.facecolor = (0.7, 0.7, 0.7)
        self.alpha = 1.0
        self.linealpha = 1.0
        self.edgewidth = 1
        self.dualedgewidth = 1

    def draw(self, fignum=None, bgcolor=None, scene_size=(900,675)):
        self.draw_mesh(fignum, bgcolor, scene_size)
        plt.show()

    def draw_mesh(self, fignum=None, bgcolor=None, scene_size=(900,675)):
        """Draw mesh. 
        
        """
        if self.edgecolor == None and self.facecolor == None:
            raise InputError('edgeolor and facecolor are both set to None')
        if self.mesh_type == 'planar':
            triangles = self.vertices[ravel(self[2].simplices),:].\
                reshape((self[2].simplices.shape[0],3,2))
            col = PolyCollection(triangles)
            col.set_linewidth(self.edgewidth)
            col.set_alpha(self.alpha)
            col.set_edgecolor(self.edgecolor)
            if self.facecolor == None :
                col.set_facecolor('none')
            else:
                col.set_facecolor(self.facecolor)
            if fignum == None:
                fig = plt.gcf()
            else:
                fig = plt.figure(fignum)
            self.figure = fig
            ax = fig.gca()
            ax.set_facecolor('none' if bgcolor is None else bgcolor)
            ax.add_collection(col, autolim = True)
            ax.autoscale_view()
            ax.set_aspect('equal')
            xmin, ymin, xmax, ymax = ax.dataLim.extents
            ax.set_xlim((xmin - (xmax-xmin)*.1, xmax + (xmax-xmin)*.1))
            ax.set_ylim((ymin - (ymax-ymin)*.1, ymax + (ymax-ymin)*.1))
            plt.show()
        elif self.mesh_type == 'surface' or self.mesh_type == 'volume':
            options = {}
            options['opacity'] = self.alpha
            options['figure'] = mmlab.figure(fignum, bgcolor, 
                                             size=scene_size)
            self.figure = options['figure']
            triangles = (self.simplices if self.mesh_type == 'surface' else 
                         self[2].simplices)
            if self.facecolor is not None:
                options['representation'] = 'surface'
                options['color'] = self.facecolor
                mmlab.triangular_mesh(self.vertices[:,0], self.vertices[:,1],
                                self.vertices[:,2], triangles,
                                **options)
            if self.edgecolor is not None:
                options['representation'] = 'wireframe'
                options['color'] = self.edgecolor
                mmlab.triangular_mesh(self.vertices[:,0], self.vertices[:,1], 
                            self.vertices[:,2], triangles, **options)


    def draw_dual_mesh(self, fignum=None, bgcolor=None,
                           scene_size=(900,675)):
        """Draw dual mesh. 
        
        """
        if self.dualedgecolor == None and self.dualfacecolor == None:
            raise InputError('edgeolor and facecolor are both set to None')
        if self.mesh_type == 'planar':
            dual_lines = set(); remove_dual_lines = set()
            for s_index, simplex in enumerate(self.simplices):
                cc_simplex = self[2].circumcenter[s_index]
                cc_simplex_bary = self[2].bary_circumcenter[s_index]
                cc_simplex_bary_signs = [0, 0, 0]
                for i in range(3):
                    if abs(cc_simplex_bary[i]) < 1e-12:
                        cc_simplex_bary_signs[i] = 0
                    else:
                        cc_simplex_bary_signs[i] = int(sign(cc_simplex_bary[i]))
                ## if ((array(cc_simplex_bary_signs) == 0).
                ##         nonzero()[0].shape[0] == 1):
                ##     print("One zero index in: ", s_index,
                ##               " ", cc_simplex_bary, cc_simplex_bary_signs)
                
                edges = [edge for edge in combinations(sort(simplex), 2)]
                local_opp_vertex = [2, 1, 0]
                edge_indices = [self[1].simplex_to_index[Simplex(edge)] for edge in edges]
                cc_edges = [self[1].circumcenter[e_index] for e_index in edge_indices]
                for e_index, cc_edge in enumerate(cc_edges):
                    cc_edge = tuple(cc_edge)
                    cc_simplex = tuple(cc_simplex)
                    # Add every half dual edge in a triangle
                    dual_lines.add((cc_edge, cc_simplex))

                    # However, if the circumcenter is outside the triangles, need to modify the
                    # addition of the half dual edge for visualization in accordance with sign
                    # convention for DEC. Thus, for a circumcenter lying outside a triangle,
                    # remove the half dual edges to the common edge of the triangle and from its
                    # neighbor where the circumcenter lies. Finish by introducing the dual edge
                    # between the circumcenters of these two triangles.
                    cc_oppvrtx_sign = cc_simplex_bary_signs[local_opp_vertex[e_index]]
                    if cc_oppvrtx_sign <= 0:
                        curr_edge_index = edge_indices[e_index]
                        opp_s_index = (set(self[1].d.getcol(curr_edge_index).indices) - {s_index})
                        if len(opp_s_index) != 0:
                            opp_s_index = opp_s_index.pop()
                            opp_cc_simplex = self[2].circumcenter[opp_s_index]                        
                            opp_cc_simplex = tuple(opp_cc_simplex)
                            dual_lines.add((cc_simplex,opp_cc_simplex))
                            remove_dual_lines.add((cc_edge, cc_simplex))
                            # There is still a catch! The triangle could be right angled and have the
                            # circumcenter on its hypotenuse. Then, have to factor this degeneracy of
                            # half dual edges while building the visualization.
                            if cc_oppvrtx_sign < 0:
                                remove_dual_lines.add((cc_edge, opp_cc_simplex))
                            else:
                                # For a right angled triangle, do not remove the edge center to
                                # circumcenter dual line since it does not exist.
                                pass
                        ## else:
                        ##     # TO DO:
                        ##     # If a boundary triangle is obtuse with
                        ##     # the circumcenter outside the simplicial
                        ##     # complex, we need to think about how to
                        ##     # handle this case
                        ##     raise NotImplementedError
                        
                            

            dual_lines -= remove_dual_lines
            dual_lines = list(dual_lines)
            col = LineCollection(dual_lines, zorder=1)
            col.set_linewidth(self.dualedgewidth)
            col.set_alpha(self.linealpha)
            col.set_edgecolor(self.dualedgecolor)
            
            if self.dualfacecolor == None :
                col.set_facecolor('none')
            else:
                col.set_facecolor(self.dualfacecolor)
            if fignum == None:
                fig = plt.gcf()
            else:
                fig = plt.figure(fignum)
            self.figure = fig
            ax = fig.gca()
            ax.set_facecolor('none' if bgcolor is None else bgcolor)
            ax.add_collection(col, autolim=True)
            ax.autoscale_view()
            ax.set_aspect('equal')
            xmin, ymin, xmax, ymax = ax.dataLim.extents
            ax.set_xlim((xmin - (xmax-xmin)*.1, xmax + (xmax-xmin)*.1))
            ax.set_ylim((ymin - (ymax-ymin)*.1, ymax + (ymax-ymin)*.1))
            plt.show()
        else:
            raise NotImplementedError('Not implemented yet!')

                

    def display_indices(self, dimensions=[0,1,2], colors=['r','g','b'],
                        fontsizes=[20,16,12], **options):
        """Display vertex, edge, triangle numbers.

        Arguments
        =========

        dimensions : list of integers selected from {0, 1, 2}. Display
        indices for which dimensions, default [0,1,2].

        colors : list of colors which will be cycled through for
        the index colors, default ['r', 'g', 'b']. Colors can be
        specified using any valid matplotlib color format, e.g. the
        default is equivalent to [(1,0,0), (0,1,0), (0,0,1)] as RGB
        triples. 

        fontsizes : list of sizes to be cycled through, default
        [20,16,12]. 

        In addition any dictionary of options that is valid for the
        matplotlib text can be passed as keyword arguments **options.

        """
        if self.mesh_type != 'planar':
            raise NotImplementedError('Indices can only be displayed ' +
                                      'for planar meshes')
        fontsize_iterator = cycle(fontsizes)
        color_iterator = cycle(colors)
        ax = self.figure.gca()

        for d in dimensions:
            if d == 0:
                vertex_color = next(color_iterator)
                vertex_fontsize = next(fontsize_iterator)
                for i,vertex in enumerate(self.vertices):
                    tx = ax.text(vertex[0], vertex[1], str(i), **options)
                    tx.set_fontsize(vertex_fontsize)
                    tx.set_color(vertex_color)
                    tx.set_horizontalalignment('center')
                    tx.set_verticalalignment('center')
            if d == 1:
                edge_color = next(color_iterator)
                edge_fontsize = next(fontsize_iterator)
                for i,edge in enumerate(self[1].simplices):
                    barycenter = average(self.vertices[edge], 0)
                    tx = ax.text(barycenter[0], barycenter[1], str(i), 
                                 **options)
                    tx.set_color(edge_color)
                    tx.set_fontsize(edge_fontsize)
                    tx.set_horizontalalignment('center')
                    tx.set_verticalalignment('center')
            if d == 2:
                triangle_color = next(color_iterator)
                triangle_fontsize = next(fontsize_iterator)
                for i,triangle in enumerate(self[2].simplices):
                    barycenter = average(self.vertices[triangle],0)
                    tx = ax.text(barycenter[0], barycenter[1], str(i),
                                 **options)
                    tx.set_color(triangle_color)
                    tx.set_fontsize(triangle_fontsize)
                    tx.set_horizontalalignment('center')
                    tx.set_verticalalignment('center')
        plt.show()
                
    def display_boundary_indices(self, dimensions=[0,1], colors=['r','g'],
                        fontsizes=[20,16], **options):
        """Display boundary vertex and edge indices.

        Arguments
        =========

        dimensions : list of integers selected from {0, 1}. Display
        indices for which dimensions, default [0,1].

        colors : list of colors which will be cycled through for
        the index colors, default ['r', 'g']. Colors can be
        specified using any valid matplotlib color format, e.g. the
        default is equivalent to [(1,0,0), (0,1,0)] as RGB
        triples.

        fontsizes : list of sizes to be cycled through, default
        [20,16]. 

        In addition any dictionary of options that is valid for the
        matplotlib text can be passed as keyword arguments **options.

        """
        if self.mesh_type != 'planar':
            raise NotImplementedError('Boundary indices can be displayed ' +
                                      'for planar meshes only')
        fontsize_iterator = cycle(fontsizes)
        color_iterator = cycle(colors)
        ax = self.figure.gca()

        boundary_edges = (self[1].d.T *
                              ones(self[2].num_simplices)).nonzero()[0]
        boundary_vertices = list(set(
            (self[1].simplices[boundary_edges]).ravel()))
        for d in dimensions:
            if d == 0:
                vertex_color = next(color_iterator)
                vertex_fontsize = next(fontsize_iterator)
                for i,vertex in zip(boundary_vertices,
                                        self.vertices[boundary_vertices]):
                    tx = ax.text(vertex[0], vertex[1], str(i), **options)
                    tx.set_fontsize(vertex_fontsize)
                    tx.set_color(vertex_color)
                    tx.set_horizontalalignment('center')
                    tx.set_verticalalignment('center')
            if d == 1:
                edge_color = next(color_iterator)
                edge_fontsize = next(fontsize_iterator)
                for i,edge in zip(boundary_edges,
                                      self[1].simplices[boundary_edges]):
                    barycenter = average(self.vertices[edge], 0)
                    tx = ax.text(barycenter[0], barycenter[1], str(i), 
                                 **options)
                    tx.set_color(edge_color)
                    tx.set_fontsize(edge_fontsize)
                    tx.set_horizontalalignment('center')
                    tx.set_verticalalignment('center')
        plt.show()

    def display_nonconnected_vertices(self, color='r',
                                    fontsize=20, **options):
        """Display user-provided vertices which are not connected to
           the user-provided simplicial complex.

        Arguments
        =========

        color : Index color; default is 'r'.

        fontsize : Default is 20.

        In addition any dictionary of options that is valid for the
        matplotlib text can be passed as keyword arguments **options.

        """
        if self.mesh_type != 'planar':
            raise NotImplementedError('Vertex indices can be displayed ' +
                                      'for planar meshes only')
        ax = self.figure.gca()

        nonconnected_vertices = list(
            set(range(self.vertices.shape[0])) -
            set(self[0].simplices.ravel()))
        for i,vertex in zip(nonconnected_vertices,
                                self.vertices[nonconnected_vertices]):
            tx = ax.text(vertex[0], vertex[1], str(i), **options)
            tx.set_fontsize(fontsize)
            tx.set_color(color)
            tx.set_horizontalalignment('center')
            tx.set_verticalalignment('bottom')
        plt.show()

    def draw_circum(self, show_centers=True, show_circles=False, **options):
        """Display circumcircles and/or circumcenters."""
        if self.complex_dimension() == 2 and self.embedding_dimension() == 2:
            N = self[2].num_simplices
            center = zeros((N,2))
            radius = zeros((N))
            for i in range(N):
                center[i],radius[i] = circumcenter(self.vertices[
                        self[2].simplices[i,:]])
            if show_centers:
                if 'markerfacecolor' in options:
                    mfc = options['markerfacecolor']
                else:
                    mfc = (1,0,0)
                if 'markersize' in options:
                    ms = options['markersize']
                else:
                    ms = 8
                if 'marker' in options:
                    marker_type = options['marker']
                else:
                    marker_type = 'o'
                plt.plot(center[:,0], center[:,1], marker=marker_type, 
                     markerfacecolor=mfc, markersize=ms, linestyle='none')
            ax = plt.gca()
            ax.autoscale_view()
            if show_circles:
                color = (0.5,0.5,0.5)
                width = None
                if 'edgecolor' in options:
                    color = options['edgecolor']
                if 'linewidth' in options:
                    width = options['linewidth']
                for i in range(N):
                    plt.gca().add_patch(Circle(center[i,:],radius[i],
                                           fc='none', ec = color,
                                           lw=width))
                xmin, ymin, xmax, ymax = ax.dataLim.extents
                ax.set_xlim((xmin - (xmax-xmin)*.1, xmax + (xmax-xmin)*.1))
                ax.set_ylim((ymin - (ymax-ymin)*.1, ymax + (ymax-ymin)*.1))
                plt.show()
        elif self.complex_dimension() == 3 and self.embedding_dimension() == 3:
            N = self[3].num_simplices; center = zeros((N, 3)); radius = zeros(N)
            for i in range(N):
                center[i], radius[i] = circumcenter(
                    self.vertices[self[3].simplices[i, :]])
            if show_centers:
                if 'mode' not in options:
                    options['mode'] = 'sphere'
                if 'color' not in options:
                    options['color'] = (0, 0, 1)
                if 'scale_factor' not in options:
                    options['scale_factor'] = 0.02
            if show_circles:
                raise NotImplementedError('Drawing of circumspheres not implemented.')
            mmlab.points3d(center[:, 0], center[:, 1], center[:, 2], **options)
        else:
            raise NotImplementedError('Drawing of circumcenters and/or ' +
                                      'circumspheres only works for ' +
                                      'planar or volume meshes')

    def draw_simplex_array(self, simplices):
        if self.mesh_type == 'planar':
            if shape(simplices)[1] == 1: # vertices
                plt.plot(self.vertices[simplices[:]], 'ro')
            else:
                pass

    def set_margin(self, margin=(0.1,0.1,0.1,0.1)):
        left, bottom, right, top = margin
        ax = self.figure.gca()
        ax.autoscale_view()
        xmin, ymin, xmax, ymax = ax.dataLim.extents
        ax.set_xlim((xmin - (xmax-xmin)*left, xmax + (xmax-xmin)*right))
        ax.set_ylim((ymin - (ymax-ymin)*bottom, ymax + (ymax-ymin)*top))
        plt.show()

    def exploded(self, fraction, fignum=None, bgcolor=None,
                 scene_size=(900,675)):
        """
        Draw an exploded view of the mesh. Each tetrahedron is moved
        by (simplex_center - mesh_center) * fraction where
        simplex_center and mesh_center are the average ove the
        vertices of the simplex and the mesh. These centers are
        computed. The rest of the behavior is as it is for
        draw_mesh().

        """
        if self.mesh_type != 'volume':
            raise NotImplementedError('Only volume meshes can be ' +
                                      'drawn exploded')
        options = {}
        options['opacity'] = self.alpha
        options['figure'] = mmlab.figure(fignum, bgcolor, 
                                         size=scene_size)
        self.figure = options['figure']        
        mesh_center = mean(self.vertices, 0)
        for s in self.simplices:
            vertices = self.vertices[s]
            simplex_center = mean(vertices, 0)
            vertices += (simplex_center - mesh_center) * fraction
            triangles = array(list(combinations(range(4),3)))
            if self.facecolor is not None:
                options['representation'] = 'surface'
                options['color'] = self.facecolor
                mmlab.triangular_mesh(vertices[:,0], vertices[:,1],
                                vertices[:,2], triangles, **options)
            if self.edgecolor is not None:
                options['representation'] = 'wireframe'
                options['color'] = self.edgecolor
                mmlab.triangular_mesh(vertices[:,0], vertices[:,1], 
                                vertices[:,2], triangles, **options)

            
class Error(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        Error.__init__(self, message)


def meshqv(v='vertices.txt', t='triangles.txt', zero=True):
    # Quick view of mesh
    vertices = loadtxt(v, dtype=float)
    triangles = loadtxt(t, dtype=int)    
    if not zero:
        triangles = triangles - 1
    vm = visualizable_simplicial_mesh(vertices, triangles)
    plt.figure()
    vm.draw()
    return vm
    

