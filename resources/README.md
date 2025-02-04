## Resources used to split Super Mario Bros. levels into succcessive "scenes"

- Each level map was first obtained from [NesMaps](https://nesmaps.com/maps/SuperMarioBrothers/SuperMarioBrothers.html) and added to the mario_scenes_manual_annotations.pptx file.
- Then for each scenes, we identified game design patterns as described in [Dahlskog & Togelius, 2012](https://doi.org/10.1145/2427116.2427117).
- Using the gym-retro GUI, we obtained the X positions corresponding to the start and end of each scene. 
- Finally, we aggregated all these informations in the scenes_mastersheet.tsv file.

**Note : **Underwater and Castle levels were ignored in our analysis because they have a slightly different gameplay than the regular level, and make use of different game design patterns.

### The scenes mastersheet

This TSV file contains informations related to all the scenes identified in Super Mario Bros. levels.


It contains one row per scene, 3 columns to identify the scene, an Entry and Exit point columns, and one columns per pattern.

These columns contain the following information : 

- World : The world ID, an integer between 1 and 8.
- Level : The level ID, an integer between 1 and 3.
- Scene : The scene ID, an integer.
- Entry point : The X position corresponding to the beginning of the scene. An integer.
- Exit point : The X position corresponding to the ending of the scene. An integer.

Design patterns (from Dahlskog & Togelius 2012). The values can be 0 (absence of the corresponding pattern) or 1 (presence of the corresponding pattern) : 
- Enemy : A single enemy
- 2-Horde : Two enemies together
- 3-Horde : Three enemies together
- 4-Horde : Four enemies together
- Roof : Enemies underneath a hanging platform making Mario bounce in the ceiling
- Gap : Single gap in the ground/platform
- Multiple gaps : More than one gap with fixed platforms in between
- Variable gaps	: Gap and platform width is variable
- Gap enemy : Enemies in the air above gaps
- Pillar gap : Pillar (pipes or blocks) are placed on platforms between gaps
- Valley : A valley created by using vertically stacked blocks or pipes but without Piranha plant(s)
- Pipe valley : A valley with pipes and Piranha plant(s)
- Empty valley : A valley without enemies
- Enemy valley : A valley with enemies
- Roof valley : A valley with enemies and a roof making Mario bounce in the ceiling
- 2-Path : A hanging platform allowing Mario to choose different paths
- 3-Path : 2 hanging platforms allowing Mario to choose different paths
- Risk/Reward : A multiple path where one path have a reward and a gap or enemy making it risky to go for the reward
- Stair up : A stair going up
- Stair down : A stair going down
- Empty stair valley : A valley between a stair up and a stair down without enemies
- Enemy stair valley : A valley between a stair up and a stair down with enemies
- Gap stair valley : A valley between a stair up and a stair down with gap in the middle

We added several patterns in order to annotate key sections of the level : 

- Reward : Rewards without immediate danger
- Moving platform : Platform moving vertically or horizontally
- Flagpole : End of the level
- Beginning : Beginning of the level
- Bonus zone : Hidden zone without enemies
- Waterworld : A special hidden zone with Waterworld gameplay