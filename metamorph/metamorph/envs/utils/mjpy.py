import itertools

from lxml import etree
import mujoco


class MjSimData:
    """Wrapper for MjData to provide backward compatibility with mujoco-py API."""

    # mujoco-py name -> mujoco 3.x name
    _ATTR_MAP = {
        "body_xpos": "xpos",
        "body_xmat": "xmat",
        "body_xquat": "xquat",
    }

    def __init__(self, model, data):
        self._model = model
        self._data = data

    def __getattr__(self, name):
        """Delegate attribute access to the underlying MjData."""
        # Handle renamed body velocity arrays
        if name == "body_xvelp":
            return self._data.cvel[:, 3:]
        if name == "body_xvelr":
            return self._data.cvel[:, :3]
        mapped = self._ATTR_MAP.get(name, name)
        return getattr(self._data, mapped)

    def get_body_xpos(self, name):
        """Get body position by name (mujoco-py compatibility)."""
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self._data.xpos[body_id]

    def get_body_xmat(self, name):
        """Get body rotation matrix by name (mujoco-py compatibility)."""
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self._data.xmat[body_id]

    def get_site_xpos(self, name):
        """Get site position by name (mujoco-py compatibility)."""
        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self._data.site_xpos[site_id]

    def get_geom_xpos(self, name):
        """Get geom position by name (mujoco-py compatibility)."""
        geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self._data.geom_xpos[geom_id]

    def get_geom_xmat(self, name):
        """Get geom rotation matrix by name (mujoco-py compatibility)."""
        geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return self._data.geom_xmat[geom_id]


class MjSim:
    """Wrapper class for backward compatibility with mujoco-py.

    In the new mujoco API, model and data are separate. This class wraps them
    to provide a similar interface to the old MjSim class.
    """

    def __init__(self, model, data=None):
        self.model = model
        self._data_raw = data if data is not None else mujoco.MjData(model)
        # Wrap data to provide backward-compatible methods
        self.data = MjSimData(model, self._data_raw)

    def step(self):
        """Advance the simulation by one step."""
        mujoco.mj_step(self.model, self._data_raw)

    def forward(self):
        """Compute forward kinematics."""
        mujoco.mj_forward(self.model, self._data_raw)

    def get_state(self):
        """Get the current simulation state."""
        return MjSimState(
            time=self._data_raw.time,
            qpos=self._data_raw.qpos.copy(),
            qvel=self._data_raw.qvel.copy(),
            act=self._data_raw.act.copy() if self.model.na > 0 else None,
            udd_state={}  # Not used in new mujoco
        )

    def set_state(self, state):
        """Set the simulation state."""
        self._data_raw.time = state.time
        self._data_raw.qpos[:] = state.qpos
        self._data_raw.qvel[:] = state.qvel
        if state.act is not None and self.model.na > 0:
            self._data_raw.act[:] = state.act


class MjSimState:
    """State container for backward compatibility."""

    def __init__(self, time, qpos, qvel, act, udd_state):
        self.time = time
        self.qpos = qpos
        self.qvel = qvel
        self.act = act
        self.udd_state = udd_state


def mj_name2id(sim, type_, name):
    """Returns the mujoco id corresponding to name."""
    if type_ == "site":
        return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, name)
    elif type_ == "geom":
        return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    elif type_ == "body":
        return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
    elif type_ == "sensor":
        return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    else:
        raise ValueError("type_ {} is not supported.".format(type_))


def mj_id2name(sim, type_, id_):
    """Returns the mujoco name corresponding to id."""
    if type_ == "site":
        return mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_SITE, id_)
    elif type_ == "geom":
        return mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, id_)
    elif type_ == "body":
        return mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, id_)
    elif type_ == "sensor":
        return mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_SENSOR, id_)
    else:
        raise ValueError("type_ {} is not supported.".format(type_))


def mjsim_from_etree(root):
    """Return MjSim from etree root."""
    model = mjmodel_from_etree(root)
    return MjSim(model)


def mjmodel_from_etree(root):
    """Return MjModel from etree root."""
    model_string = etree.tostring(root, encoding="unicode", pretty_print=True)
    return mujoco.MjModel.from_xml_string(model_string)


def joint_qpos_idxs(sim, joint_name):
    """Gets indexes for the specified joint's qpos values.

    Args:
        sim: MjSim object with .model attribute (MjModel)
        joint_name: Name of the joint
    """
    joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    joint_type = sim.model.jnt_type[joint_id]
    qposadr = sim.model.jnt_qposadr[joint_id]

    # Different joint types have different qpos sizes
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        # Free joint: 7 values (3 position + 4 quaternion)
        return list(range(qposadr, qposadr + 7))
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        # Ball joint: 4 values (quaternion)
        return list(range(qposadr, qposadr + 4))
    else:
        # Hinge or slide joint: 1 value
        return [qposadr]


def qpos_idxs_from_joint_prefix(sim, prefix):
    """Gets indexes for the qpos values of all joints matching the prefix."""
    joint_names = [sim.model.joint(i).name for i in range(sim.model.njnt)]
    qpos_idxs_list = [
        joint_qpos_idxs(sim, name)
        for name in joint_names
        if name.startswith(prefix)
    ]
    return list(itertools.chain.from_iterable(qpos_idxs_list))


def qpos_idxs_for_agent(sim):
    """Gets indexes for the qpos values of all agent joints."""
    agent_joints = names_from_prefixes(sim, ["root", "torso", "limb"], "joint")
    qpos_idxs_list = [joint_qpos_idxs(sim, name) for name in agent_joints]
    return list(itertools.chain.from_iterable(qpos_idxs_list))


def joint_qvel_idxs(sim, joint_name):
    """Gets indexes for the specified joint's qvel values.

    Args:
        sim: MjSim object with .model attribute (MjModel)
        joint_name: Name of the joint
    """
    joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    joint_type = sim.model.jnt_type[joint_id]
    dofadr = sim.model.jnt_dofadr[joint_id]

    # Different joint types have different qvel sizes
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        # Free joint: 6 values (3 linear + 3 angular velocity)
        return list(range(dofadr, dofadr + 6))
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        # Ball joint: 3 values (angular velocity)
        return list(range(dofadr, dofadr + 3))
    else:
        # Hinge or slide joint: 1 value
        return [dofadr]


def qvel_idxs_from_joint_prefix(sim, prefix):
    """Gets indexes for the qvel values of all joints matching the prefix."""
    joint_names = [sim.model.joint(i).name for i in range(sim.model.njnt)]
    qvel_idxs_list = [
        joint_qvel_idxs(sim, name)
        for name in joint_names
        if name.startswith(prefix)
    ]
    return list(itertools.chain.from_iterable(qvel_idxs_list))


def qvel_idxs_for_agent(sim):
    """Gets indexes for the qvel values of all agent joints."""
    agent_joints = names_from_prefixes(sim, ["root", "torso", "limb"], "joint")
    qvel_idxs_list = [joint_qvel_idxs(sim, name) for name in agent_joints]
    return list(itertools.chain.from_iterable(qvel_idxs_list))


def names_from_prefixes(sim, prefixes, elem_type):
    """Get all names of elem_type elems which match any of the prefixes."""
    # Get the count and accessor for the element type
    if elem_type == "joint":
        count = sim.model.njnt
        get_name = lambda i: sim.model.joint(i).name
    elif elem_type == "body":
        count = sim.model.nbody
        get_name = lambda i: sim.model.body(i).name
    elif elem_type == "geom":
        count = sim.model.ngeom
        get_name = lambda i: sim.model.geom(i).name
    elif elem_type == "site":
        count = sim.model.nsite
        get_name = lambda i: sim.model.site(i).name
    else:
        raise ValueError(f"elem_type {elem_type} is not supported.")

    all_names = [get_name(i) for i in range(count)]
    matches = []
    for name in all_names:
        for prefix in prefixes:
            if name.startswith(prefix):
                matches.append(name)
                break
    return matches


def geom_idxs_for_agent(sim):
    """Gets indexes for agent geoms."""
    agent_geoms = names_from_prefixes(sim, ["torso", "limb"], "geom")
    geom_idx_list = [
        mj_name2id(sim, "geom", geom_name) for geom_name in agent_geoms
    ]
    return geom_idx_list


def body_idxs_for_agent(sim):
    """Gets indexes for agent body."""
    agent_bodies = names_from_prefixes(sim, ["torso", "limb"], "body")
    body_idx_list = [
        mj_name2id(sim, "body", body_name) for body_name in agent_bodies
    ]
    return body_idx_list


def get_active_contacts(sim):
    """Get active contacts in the simulation."""
    num_contacts = sim.data.ncon
    contacts = sim.data.contact[:num_contacts]
    contact_geoms = [
        tuple(
            sorted(
                (
                    mj_id2name(sim, "geom", contact.geom1),
                    mj_id2name(sim, "geom", contact.geom2),
                )
            )
        )
        for contact in contacts
    ]
    return sorted(list(set(contact_geoms)))
