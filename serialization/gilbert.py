# ------------------------------------------------------------------
"""
Gilbert (generalized Hilbert) space-filling curve

Based on:
    https://github.com/jakubcerveny/gilbert/blob/master/gilbert_d2xyz.py
    SPDX-License-Identifier: BSD-2-Clause
    Copyright (c) 2024 abetusk

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

def gilbert_d2xy(idx, w, h):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Takes a position along the gilbert curve and returns
    its 2D (x,y) coordinate.
    """

    if w >= h:
        return gilbert_d2xy_r(idx,0, 0,0, w,0, 0,h)
    return gilbert_d2xy_r(idx,0, 0,0, 0,h, w,0)

def gilbert_d2xy_r(dst_idx, cur_idx, x,y, ax,ay, bx,by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay))  # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by))  # unit orthogonal direction

    #dx = dax + dbx
    #dy = day + dby
    di = dst_idx - cur_idx

    if h == 1: return (x + dax*di, y + day*di)
    if w == 1: return (x + dbx*di, y + dby*di)

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        nxt_idx = cur_idx + abs((ax2 + ay2)*(bx + by))
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xy_r(dst_idx, cur_idx,  x, y, ax2, ay2, bx, by)
        cur_idx = nxt_idx

        return gilbert_d2xy_r(dst_idx, cur_idx, x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    if (h2 % 2) and (h > 2):
        # prefer even steps
        (bx2, by2) = (bx2 + dbx, by2 + dby)

    # standard case: one step up, one long horizontal, one step down
    nxt_idx = cur_idx + abs((bx2 + by2)*(ax2 + ay2))
    if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
        return gilbert_d2xy_r(dst_idx, cur_idx, x,y, bx2,by2, ax2,ay2)
    cur_idx = nxt_idx

    nxt_idx = cur_idx + abs((ax + ay)*((bx - bx2) + (by - by2)))
    if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
        return gilbert_d2xy_r(dst_idx, cur_idx, x+bx2, y+by2, ax,ay, bx-bx2,by-by2)
    cur_idx = nxt_idx

    return gilbert_d2xy_r(dst_idx, cur_idx,
                          x+(ax-dax)+(bx2-dbx),
                          y+(ay-day)+(by2-dby),
                          -bx2, -by2,
                          -(ax-ax2), -(ay-ay2))

def gilbert_xy2d(x, y, w, h):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Takes a discrete 2D coordinate and maps it to the
    index position on the gilbert curve.
    """

    if w >= h:
        return gilbert_xy2d_r(0, x,y, 0,0, w,0, 0,h)
    return gilbert_xy2d_r(0, x,y, 0,0, 0,h, w,0)

def in_bounds_2d(x, y, x_s, y_s, ax, ay, bx, by):

    dx = ax + bx
    dy = ay + by

    if dx < 0:
        if (x > x_s) or (x <= (x_s + dx)): return False
    else:
        if (x < x_s) or (x >= (x_s + dx)): return False

    if dy < 0:
        if (y > y_s) or (y <= (y_s + dy)): return False
    else:
        if (y < y_s) or (y >= (y_s + dy)): return False

    return True

def gilbert_xy2d_r(cur_idx, x_dst, y_dst, x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay))  # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by))  # unit orthogonal direction

    dx = dax + dbx
    dy = day + dby

    if h == 1:
        if (dax==0): return cur_idx + (dy*(y_dst - y))
        return cur_idx + (dx*(x_dst - x))

    if w == 1:
        if (dbx==0): return cur_idx + (dy*(y_dst - y))
        return cur_idx + (dx*(x_dst - x))

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        if in_bounds_2d(x_dst, y_dst, x, y, ax2, ay2, bx, by):
            return gilbert_xy2d_r(cur_idx, x_dst, y_dst, x, y, ax2, ay2, bx, by)

        cur_idx += abs((ax2 + ay2)*(bx + by))
        return gilbert_xy2d_r(cur_idx, x_dst, y_dst, x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        if in_bounds_2d(x_dst, y_dst, x, y, bx2, by2, ax2, ay2):
            return gilbert_xy2d_r(cur_idx, x_dst,y_dst, x,y, bx2,by2, ax2,ay2)
        cur_idx += abs((bx2 + by2)*(ax2 + ay2))

        if in_bounds_2d(x_dst, y_dst, x+bx2, y+by2, ax, ay, bx-bx2, by-by2):
            return gilbert_xy2d_r(cur_idx, x_dst,y_dst, x+bx2,y+by2, ax,ay, bx-bx2,by-by2)
        cur_idx += abs((ax + ay)*((bx - bx2) + (by - by2)))

        return gilbert_xy2d_r(cur_idx, x_dst,y_dst,
                              x+(ax-dax)+(bx2-dbx),
                              y+(ay-day)+(by2-dby),
                              -bx2, -by2,
                              -(ax-ax2), -(ay-ay2))


def gilbert_d2xyz(idx, width, height, depth):
    """
    Generalized Hilbert ('Gilbert') space-filling curve for arbitrary-sized
    3D rectangular grids. Generates discrete 3D coordinates to fill a cuboid
    of size (width x height x depth). Even sizes are recommended in 3D.
    """

    if width >= height and width >= depth:
       return gilbert_d2xyz_r(idx, 0,
                              0, 0, 0,
                              width, 0, 0,
                              0, height, 0,
                              0, 0, depth)

    elif height >= width and height >= depth:
       return gilbert_d2xyz_r(idx, 0,
                              0, 0, 0,
                              0, height, 0,
                              width, 0, 0,
                              0, 0, depth)

    else: # depth >= width and depth >= height
       return gilbert_d2xyz_r(idx, 0,
                              0, 0, 0,
                              0, 0, depth,
                              width, 0, 0,
                              0, height, 0)

def gilbert_d2xyz_r(dst_idx, cur_idx,
                    x, y, z,
                    ax, ay, az,
                    bx, by, bz,
                    cx, cy, cz):

    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az))  # unit major direction ("right")
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz))  # unit ortho direction ("forward")
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz))  # unit ortho direction ("up")

    _dx = dax + dbx + dcx
    _dy = day + dby + dcy
    _dz = daz + dbz + dcz
    _di = dst_idx - cur_idx

    # trivial row/column fills
    if h == 1 and d == 1:
        return (x + dax*_di, y + day*_di, z + daz*_di)

    if w == 1 and d == 1:
        return (x + dbx*_di, y + dby*_di, z + dbz*_di)

    if w == 1 and h == 1:
        return (x + dcx*_di, y + dcy*_di, z + dcz*_di)

    (ax2, ay2, az2) = (ax//2, ay//2, az//2)
    (bx2, by2, bz2) = (bx//2, by//2, bz//2)
    (cx2, cy2, cz2) = (cx//2, cy//2, cz//2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # prefer even steps
    if (w2 % 2) and (w > 2): (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)
    if (h2 % 2) and (h > 2): (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)
    if (d2 % 2) and (d > 2): (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    # wide case, split in w only
    if (2*w > 3*h) and (2*w > 3*d):
        nxt_idx = cur_idx + abs( (ax2 + ay2 + az2)*(bx + by + bz)*(cx + cy + cz) )
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xyz_r(dst_idx,cur_idx,
                                   x, y, z,
                                   ax2, ay2, az2,
                                   bx, by, bz,
                                   cx, cy, cz)
        cur_idx = nxt_idx

        return gilbert_d2xyz_r(dst_idx,cur_idx,
                               x+ax2, y+ay2, z+az2,
                               ax-ax2, ay-ay2, az-az2,
                               bx, by, bz,
                               cx, cy, cz)

    # do not split in d
    elif 3*h > 4*d:
        nxt_idx = cur_idx + abs( (bx2 + by2 + bz2)*(cx + cy + cz)*(ax2 + ay2 + az2) )
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xyz_r(dst_idx,cur_idx,
                                   x, y, z,
                                   bx2, by2, bz2,
                                   cx, cy, cz,
                                   ax2, ay2, az2)
        cur_idx = nxt_idx

        nxt_idx = cur_idx + abs( (ax + ay + az)*((bx - bx2) + (by - by2) + (bz - bz2))*(cx + cy + cz) )
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xyz_r(dst_idx,cur_idx,
                                   x+bx2, y+by2, z+bz2,
                                   ax, ay, az,
                                   bx-bx2, by-by2, bz-bz2,
                                   cx, cy, cz)
        cur_idx = nxt_idx

        return gilbert_d2xyz_r(dst_idx,cur_idx,
                               x+(ax-dax)+(bx2-dbx),
                               y+(ay-day)+(by2-dby),
                               z+(az-daz)+(bz2-dbz),
                               -bx2, -by2, -bz2,
                               cx, cy, cz,
                               -(ax-ax2), -(ay-ay2), -(az-az2))

    # do not split in h
    elif 3*d > 4*h:
        nxt_idx = cur_idx + abs( (cx2 + cy2 + cz2)*(ax2 + ay2 + az2)*(bx + by + bz) )
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xyz_r(dst_idx,cur_idx,
                                   x, y, z,
                                   cx2, cy2, cz2,
                                   ax2, ay2, az2,
                                   bx, by, bz)
        cur_idx = nxt_idx

        nxt_idx = cur_idx + abs( (ax + ay + az)*(bx + by + bz)*((cx - cx2) + (cy - cy2) + (cz - cz2)) )
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xyz_r(dst_idx,cur_idx,
                                   x+cx2, y+cy2, z+cz2,
                                   ax, ay, az,
                                   bx, by, bz,
                                   cx-cx2, cy-cy2, cz-cz2)
        cur_idx = nxt_idx

        return gilbert_d2xyz_r(dst_idx,cur_idx,
                               x+(ax-dax)+(cx2-dcx),
                               y+(ay-day)+(cy2-dcy),
                               z+(az-daz)+(cz2-dcz),
                               -cx2, -cy2, -cz2,
                               -(ax-ax2), -(ay-ay2), -(az-az2),
                               bx, by, bz)

    # regular case, split in all w/h/d
    else:
        nxt_idx = cur_idx + abs( (bx2 + by2 + bz2)*(cx2 + cy2 + cz2)*(ax2 + ay2 + az2) )
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xyz_r(dst_idx,cur_idx,
                                   x, y, z,
                                   bx2, by2, bz2,
                                   cx2, cy2, cz2,
                                   ax2, ay2, az2)
        cur_idx = nxt_idx

        nxt_idx = cur_idx + abs( (cx + cy + cz)*(ax2 + ay2 + az2)*((bx - bx2) + (by - by2) + (bz - bz2)) )
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xyz_r(dst_idx,cur_idx,
                                   x+bx2, y+by2, z+bz2,
                                   cx, cy, cz,
                                   ax2, ay2, az2,
                                   bx-bx2, by-by2, bz-bz2)
        cur_idx = nxt_idx

        nxt_idx = cur_idx + abs( (ax + ay + az)*(-bx2 - by2 - bz2)*(-(cx - cx2) - (cy - cy2) - (cz - cz2)) )
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xyz_r(dst_idx, cur_idx,
                                   x+(bx2-dbx)+(cx-dcx),
                                   y+(by2-dby)+(cy-dcy),
                                   z+(bz2-dbz)+(cz-dcz),
                                   ax, ay, az,
                                   -bx2, -by2, -bz2,
                                   -(cx-cx2), -(cy-cy2), -(cz-cz2))
        cur_idx = nxt_idx

        nxt_idx = cur_idx + abs( (-cx - cy - cz)*(-(ax - ax2) - (ay - ay2) - (az - az2))*((bx - bx2) + (by - by2) + (bz - bz2)) )
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xyz_r(dst_idx,cur_idx,
                                   x+(ax-dax)+bx2+(cx-dcx),
                                   y+(ay-day)+by2+(cy-dcy),
                                   z+(az-daz)+bz2+(cz-dcz),
                                   -cx, -cy, -cz,
                                   -(ax-ax2), -(ay-ay2), -(az-az2),
                                   bx-bx2, by-by2, bz-bz2)
        cur_idx = nxt_idx

        return gilbert_d2xyz_r(dst_idx,cur_idx,
                               x+(ax-dax)+(bx2-dbx),
                               y+(ay-day)+(by2-dby),
                               z+(az-daz)+(bz2-dbz),
                               -bx2, -by2, -bz2,
                               cx2, cy2, cz2,
                               -(ax-ax2), -(ay-ay2), -(az-az2))


def gilbert_xyz2d(x, y, z, width, height, depth):
    """
    Generalized Hilbert ('Gilbert') space-filling curve for arbitrary-sized
    3D rectangular grids. Generates discrete 3D coordinates to fill a cuboid
    of size (width x height x depth). Even sizes are recommended in 3D.
    """

    if width >= height and width >= depth:
       return gilbert_xyz2d_r(0,x,y,z,
                              0, 0, 0,
                              width, 0, 0,
                              0, height, 0,
                              0, 0, depth)

    elif height >= width and height >= depth:
       return gilbert_xyz2d_r(0,x,y,z,
                              0, 0, 0,
                              0, height, 0,
                              width, 0, 0,
                              0, 0, depth)

    else: # depth >= width and depth >= height
       return gilbert_xyz2d_r(0,x,y,z,
                              0, 0, 0,
                              0, 0, depth,
                              width, 0, 0,
                              0, height, 0)


def in_bounds_3d(x, y, z, x_s, y_s, z_s, ax, ay, az, bx, by, bz, cx, cy, cz):

    dx = ax + bx + cx
    dy = ay + by + cy
    dz = az + bz + cz

    if dx < 0:
        if (x > x_s) or (x <= (x_s + dx)): return False
    else:
        if (x < x_s) or (x >= (x_s + dx)): return False

    if dy < 0:
        if (y > y_s) or (y <= (y_s + dy)): return False
    else:
        if (y < y_s) or (y >= (y_s + dy)): return False

    if dz <0:
        if (z > z_s) or (z <= (z_s + dz)): return False
    else:
        if (z < z_s) or (z >= (z_s + dz)): return False

    return True


def gilbert_xyz2d_r(cur_idx,
                    x_dst,y_dst,z_dst,
                    x, y, z,
                    ax, ay, az,
                    bx, by, bz,
                    cx, cy, cz):

    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az))  # unit major direction ("right")
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz))  # unit ortho direction ("forward")
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz))  # unit ortho direction ("up")

    # trivial row/column fills
    if h == 1 and d == 1:
        return cur_idx + (dax*(x_dst - x)) + (day*(y_dst - y)) + (daz*(z_dst - z))

    if w == 1 and d == 1:
        return cur_idx + (dbx*(x_dst - x)) + (dby*(y_dst - y)) + (dbz*(z_dst - z))

    if w == 1 and h == 1:
        return cur_idx + (dcx*(x_dst - x)) + (dcy*(y_dst - y)) + (dcz*(z_dst - z))

    (ax2, ay2, az2) = (ax//2, ay//2, az//2)
    (bx2, by2, bz2) = (bx//2, by//2, bz//2)
    (cx2, cy2, cz2) = (cx//2, cy//2, cz//2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # prefer even steps
    if (w2 % 2) and (w > 2):
       (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)

    if (h2 % 2) and (h > 2):
       (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)

    if (d2 % 2) and (d > 2):
       (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    # wide case, split in w only
    if (2*w > 3*h) and (2*w > 3*d):
        if in_bounds_3d(x_dst,y_dst,z_dst,
                     x,y,z,
                     ax2,ay2,az2,
                     bx,by,bz,
                     cx,cy,cz):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x, y, z,
                                   ax2, ay2, az2,
                                   bx, by, bz,
                                   cx, cy, cz)
        cur_idx += abs( (ax2 + ay2 + az2)*(bx + by + bz)*(cx + cy + cz) )

        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+ax2, y+ay2, z+az2,
                               ax-ax2, ay-ay2, az-az2,
                               bx, by, bz,
                               cx, cy, cz)

    # do not split in d
    elif 3*h > 4*d:
        if in_bounds_3d(x_dst,y_dst,z_dst,
                     x,y,z,
                     bx2,by2,bz2,
                     cx,cy,cz,
                     ax2,ay2,az2):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x, y, z,
                                   bx2, by2, bz2,
                                   cx, cy, cz,
                                   ax2, ay2, az2)
        cur_idx += abs( (bx2 + by2 + bz2)*(cx + cy + cz)*(ax2 + ay2 + az2) )

        if in_bounds_3d(x_dst,y_dst,z_dst,
                     x+bx2,y+by2,z+bz2,
                     ax,ay,az,
                     bx-bx2,by-by2,bz-bz2,
                     cx,cy,cz):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x+bx2, y+by2, z+bz2,
                                   ax, ay, az,
                                   bx-bx2, by-by2, bz-bz2,
                                   cx, cy, cz)
        cur_idx += abs( (ax + ay + az)*((bx - bx2) + (by - by2) + (bz - bz2))*(cx + cy + cz) )

        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(ax-dax)+(bx2-dbx),
                               y+(ay-day)+(by2-dby),
                               z+(az-daz)+(bz2-dbz),
                               -bx2, -by2, -bz2,
                               cx, cy, cz,
                               -(ax-ax2), -(ay-ay2), -(az-az2))

    # do not split in h
    elif 3*d > 4*h:
        if in_bounds_3d(x_dst,y_dst,z_dst,
                     x,y,z,
                     cx2,cy2,cz2,
                     ax2,ay2,az2, bx,by,bz):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x, y, z,
                                   cx2, cy2, cz2,
                                   ax2, ay2, az2,
                                   bx, by, bz)
        cur_idx += abs( (cx2 + cy2 + cz2)*(ax2 + ay2 + az2)*(bx + by + bz) )

        if in_bounds_3d(x_dst,y_dst,z_dst,
                     x+cx2,y+cy2,z+cz2,
                     ax,ay,az, bx,by,bz,
                     cx-cx2,cy-cy2,cz-cz2):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x+cx2, y+cy2, z+cz2,
                                   ax, ay, az,
                                   bx, by, bz,
                                   cx-cx2, cy-cy2, cz-cz2)
        cur_idx += abs( (ax + ay + az)*(bx + by + bz)*((cx - cx2) + (cy - cy2) + (cz - cz2)) )

        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(ax-dax)+(cx2-dcx),
                               y+(ay-day)+(cy2-dcy),
                               z+(az-daz)+(cz2-dcz),
                               -cx2, -cy2, -cz2,
                               -(ax-ax2), -(ay-ay2), -(az-az2),
                               bx, by, bz)

    # regular case, split in all w/h/d
    if in_bounds_3d(x_dst,y_dst,z_dst,
                 x,y,z,
                 bx2,by2,bz2,
                 cx2,cy2,cz2,
                 ax2,ay2,az2):
        return gilbert_xyz2d_r(cur_idx,x_dst,y_dst,z_dst,
                              x, y, z,
                              bx2, by2, bz2,
                              cx2, cy2, cz2,
                              ax2, ay2, az2)
    cur_idx += abs( (bx2 + by2 + bz2)*(cx2 + cy2 + cz2)*(ax2 + ay2 + az2) )

    if in_bounds_3d(x_dst,y_dst,z_dst,
                 x+bx2, y+by2, z+bz2,
                 cx, cy, cz,
                 ax2, ay2, az2,
                 bx-bx2, by-by2, bz-bz2):
        return gilbert_xyz2d_r(cur_idx,
                              x_dst,y_dst,z_dst,
                              x+bx2, y+by2, z+bz2,
                              cx, cy, cz,
                              ax2, ay2, az2,
                              bx-bx2, by-by2, bz-bz2)
    cur_idx += abs( (cx + cy + cz)*(ax2 + ay2 + az2)*((bx - bx2) + (by - by2) + (bz - bz2)) )

    if in_bounds_3d(x_dst,y_dst,z_dst,
                 x+(bx2-dbx)+(cx-dcx),
                 y+(by2-dby)+(cy-dcy),
                 z+(bz2-dbz)+(cz-dcz),
                 ax, ay, az,
                 -bx2, -by2, -bz2,
                 -(cx-cx2), -(cy-cy2), -(cz-cz2)):
        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(bx2-dbx)+(cx-dcx),
                               y+(by2-dby)+(cy-dcy),
                               z+(bz2-dbz)+(cz-dcz),
                               ax, ay, az,
                               -bx2, -by2, -bz2,
                               -(cx-cx2), -(cy-cy2), -(cz-cz2))
    cur_idx += abs( (ax + ay + az)*(-bx2 - by2 - bz2)*(-(cx - cx2) - (cy - cy2) - (cz - cz2)) )

    if in_bounds_3d(x_dst,y_dst,z_dst,
                 x+(ax-dax)+bx2+(cx-dcx),
                 y+(ay-day)+by2+(cy-dcy),
                 z+(az-daz)+bz2+(cz-dcz),
                 -cx, -cy, -cz,
                 -(ax-ax2), -(ay-ay2), -(az-az2),
                 bx-bx2, by-by2, bz-bz2):
        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(ax-dax)+bx2+(cx-dcx),
                               y+(ay-day)+by2+(cy-dcy),
                               z+(az-daz)+bz2+(cz-dcz),
                               -cx, -cy, -cz,
                               -(ax-ax2), -(ay-ay2), -(az-az2),
                               bx-bx2, by-by2, bz-bz2)
    cur_idx += abs( (-cx - cy - cz)*(-(ax - ax2) - (ay - ay2) - (az - az2))*((bx - bx2) + (by - by2) + (bz - bz2)) )

    return gilbert_xyz2d_r(cur_idx,
                           x_dst,y_dst,z_dst,
                           x+(ax-dax)+(bx2-dbx),
                           y+(ay-day)+(by2-dby),
                           z+(az-daz)+(bz2-dbz),
                           -bx2, -by2, -bz2,
                           cx2, cy2, cz2,
                           -(ax-ax2), -(ay-ay2), -(az-az2))


if __name__ == "__main__":

    width = 45  # 7200  # 45
    height = 90  # 3000  # 90

    xs, ys = [], []
    xst, yst = [], []

    from itertools import product
    import numpy as np

    for idx in range(width*height):
        (x, y) = gilbert_d2xy(idx, width, height)
        xs.append(x)
        ys.append(y)
        yst.append(-y+height-1)

    indices = []

    for x, y in product(np.arange(width), np.arange(height)):
        # Trans_Gilbert
        #idx = gilbert_xy2d(x, -y+height-1, width, height)
        idx = gilbert_xy2d(x, y, width, height)

        indices.append(idx)

    #np.save(r'/home/ssd4tb/shams/GloFAS_Static/gilbert_trans_test.npy',
    #        np.array(indices).reshape(width, height).T.astype(np.uint32))

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('TkAgg')
    #plt.figure(figsize=(12, 6))
    plt.plot(xs, ys, '.-')
    #plt.plot(xs, yst, '.-', color='red', alpha=0.5)
    plt.show()

