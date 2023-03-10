#################################################### 
# Credits of some functions to -> Juan Gallostra Acín's repository - https://github.com/juangallostra/augmented-reality 
#####################################################

'''
Copyright (c) 2018 Juan Gallostra Acín

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import cv2
import numpy as np


def augment(img, obj, projection, template, scale = 4):
    # takes the captureed image, object to augment, and transformation matrix  
    #adjust scale to make the object smaller or bigger, 4 works for object

    h, w = template.shape
    vertices = obj.vertices
    img = np.ascontiguousarray(img, dtype=np.uint8)

    # Blacked the aruco marker
    a = np.array([[0,0,0], [w, 0, 0],  [w,h,0],  [0, h, 0]], np.float64 )
    imgpts = np.int32(cv2.perspectiveTransform(a.reshape(-1, 1, 3), projection))
    cv2.fillConvexPoly(img, imgpts, (0,0,0))

    # Projection of the faces to pixel coords and then drawing
    for face in obj.faces:
        # Face is list of [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices]) # vertex -1 -> shifted numbering
        points = scale*points
        points = np.array([[p[2] + w/2, p[0] + h/2, p[1]] for p in points]) # displaced to centre
	# Transforming to pixel coords 
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])
        
    return img

class find_3d_object:
    def __init__(self, filename_obj, filename_texture, color_fixed = False):
        self.texture = cv2.imread(filename_texture)
        self.vertices = []
        self.faces = []
        #each face is a list of [lis_vertices, lis_texcoords, color]
        self.texcoords = []

        for line in open(filename_obj, "r"):
            if line.startswith('#'): 
                continue

            values = line.split()
            if not values:
                continue
            
            if values[0] == 'v':
                # Vertex (x, y, z)
                v = [float(a) for a in values[1:4] ]
                self.vertices.append(v)

            elif values[0] == 'vt':
                # Texture coordinate (u, v)
                self.texcoords.append([float(a) for a in values[1:3] ])

            elif values[0] == 'f':
                # Face Description 
                face_vertices = []
                face_texcoords = []
                for v in values[1:]:
                    w = v.split('/')
                    face_vertices.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        face_texcoords.append(int(w[1]))
                    else:
                        color_fixed = True
                        face_texcoords.append(0)
                self.faces.append([face_vertices, face_texcoords])


        for f in self.faces:
            if not color_fixed:
                f.append(find_3d_object.decide_face_color(f[-1], self.texture, self.texcoords))
            else:
                f.append((50, 50, 50)) # Default color

        # cv2.imwrite('texture_marked.png', self.texture)

    def decide_face_color(hex_color, texture, textures):
        #doesnt use proper texture
        #takes the color at the mean of the texture coords

        h, w, _ = texture.shape
        col = np.zeros(3)
        coord = np.zeros(2)
        all_us = []
        all_vs = []

        for i in hex_color:
            t = textures[i - 1]
            coord = np.array([t[0], t[1]])
            u , v = int(w*(t[0]) - 0.0001), int(h*(1-t[1])- 0.0001)
            all_us.append(u)
            all_vs.append(v)

        u = int(sum(all_us)/len(all_us))
        v = int(sum(all_vs)/len(all_vs))

        # all_us.append(all_us[0])
        # all_vs.append(all_vs[0])
        # for i in range(len(all_us) - 1):
        #     texture = cv2.line(texture, (all_us[i], all_vs[i]), (all_us[i + 1], all_vs[i + 1]), (0,0,255), 2)
        #     pass    

        col = np.uint8(texture[v, u])
        col = [int(a) for a in col]
        col = tuple(col)
        return (col)
