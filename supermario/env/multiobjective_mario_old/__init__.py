from gym.envs.registration import register
from .package_info import USERNAME
from .nes_env import NesEnv, MetaNesEnv
from .super_mario_bros import SuperMarioBrosEnv, MetaSuperMarioBrosEnv

# Env registration
# ==========================
# (world_number, level_number, area_number, max_distance)
SMB_LEVELS = [
    (1, 1, 1, 3266), (1, 2, 3, 3266), (1, 3, 4, 2514), (1, 4, 5, 2430),
    (2, 1, 1, 3298), (2, 2, 3, 3266), (2, 3, 4, 3682), (2, 4, 5, 2430),
    (3, 1, 1, 3298), (3, 2, 2, 3442), (3, 3, 3, 2498), (3, 4, 4, 2430),
    (4, 1, 1, 3698), (4, 2, 3, 3266), (4, 3, 4, 2434), (4, 4, 5, 2942),
    (5, 1, 1, 3282), (5, 2, 2, 3298), (5, 3, 3, 2514), (5, 4, 4, 2429),
    (6, 1, 1, 3106), (6, 2, 2, 3554), (6, 3, 3, 2754), (6, 4, 4, 2429),
    (7, 1, 1, 2962), (7, 2, 3, 3266), (7, 3, 4, 3682), (7, 4, 5, 3453),
    (8, 1, 1, 6114), (8, 2, 2, 3554), (8, 3, 3, 3554), (8, 4, 4, 4989)]

for draw_tiles in range(2):
    tile_suffix = '-Tiles' if draw_tiles == 1 else ''

    register(
        id='{}/meta-SuperMarioBros{}-v0'.format(USERNAME, tile_suffix),
        entry_point='{}_gym_super_mario:MetaSuperMarioBrosEnv'.format(USERNAME),
        max_episode_steps=9999999,
        reward_threshold=32000,
        kwargs={ 'draw_tiles': draw_tiles, 'average_over': 3, 'passing_grade': 600, 'min_tries_for_avg': 3 },
        nondeterministic=True,
    )

    for (world_number, level_number, area_number, max_distance) in SMB_LEVELS:
        level = (world_number - 1) * 4 + (level_number - 1)
        register(
            id='{}/SuperMarioBros-{}-{}{}-v0'.format(USERNAME, world_number, level_number, tile_suffix),
            entry_point='{}_gym_super_mario:SuperMarioBrosEnv'.format(USERNAME),
            max_episode_steps=10000,
            reward_threshold=(max_distance - 40),
            kwargs={ 'draw_tiles': draw_tiles, 'level': level },
            # Seems to be non-deterministic about 5% of the time
            nondeterministic=True,
        )
