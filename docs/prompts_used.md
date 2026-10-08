# 已跑视频用到的 prompt

### 踏板车 头盔 → 牛仔帽，新 prompt（v3 纯描述）：baseline A / B / C / D、C v1 r2

- 编辑词：a white open-face helmet → a brown cowboy hat（SAM3：helmet）
- 源 prompt：A young man rides a beige vintage scooter on a rural road between cornfields and green hills under a cloudy sky. He wears a white open-face helmet, sunglasses, a red bandana and a white t-shirt.
- 目标 prompt：A young man rides a beige vintage scooter on a rural road between cornfields and green hills under a cloudy sky. He wears a brown cowboy hat, sunglasses, a red bandana and a white t-shirt.

### 踏板车 头盔 → 牛仔帽，旧 prompt（v2）：C_ours_gate_oldprompt；也是现在 edits.json 里的版本

- 编辑词：a white open-face helmet → a brown cowboy hat（SAM3：helmet）
- 源 prompt：A multi-shot video of a young man riding a beige vintage scooter on a rural road between cornfields and green hills under a cloudy sky. He wears a white open-face helmet, sunglasses, a red bandana and a white t-shirt. The video cuts between a close-up of his face, a medium shot while riding, a wide shot on the road, and a wide shot of him standing beside the parked scooter.
- 目标 prompt：A multi-shot video of a young man riding a beige vintage scooter on a rural road between cornfields and green hills under a cloudy sky. He wears a brown cowboy hat, sunglasses, a red bandana and a white t-shirt. The video cuts between a close-up of his face, a medium shot while riding, a wide shot on the road, and a wide shot of him standing beside the parked scooter.

### 骑车人 白T → 印花T：baseline A / B / C / D、memcheck C / M

- 编辑词：a white t-shirt → a t-shirt with a large colorful graphic print（SAM3：white t-shirt）
- 源 prompt：A young man rides a black fixed-gear bicycle through city streets and crosswalks among cars and buildings. He wears a white t-shirt, black pants and black sneakers.
- 目标 prompt：A young man rides a black fixed-gear bicycle through city streets and crosswalks among cars and buildings. He wears a t-shirt with a large colorful graphic print, black pants and black sneakers.

### 玩偶 黄裙 → 蓝裙，带 cup：baseline A / B / C / D、E0 pilot（已停）

- 编辑词：a yellow dress → a blue dress（SAM3：yellow dress）
- 源 prompt：A children's puppet show in a colorful room with bow-patterned wallpaper. A girl with a black bob haircut wearing a yellow dress stands beside a brown cat puppet with round glasses holding a green cup, while a man in a purple shirt and white striped pants talks to them.
- 目标 prompt：A children's puppet show in a colorful room with bow-patterned wallpaper. A girl with a black bob haircut wearing a blue dress stands beside a brown cat puppet with round glasses holding a green cup, while a man in a purple shirt and white striped pants talks to them.

### 玩偶 黄裙 → 蓝裙，去掉 cup：A_stock_nocup、C nocup、C 不膨胀、C nogap

- 编辑词：a yellow dress → a blue dress（SAM3：yellow dress）
- 源 prompt：A children's puppet show in a colorful room with bow-patterned wallpaper. A girl with a black bob haircut wearing a yellow dress stands beside a brown cat puppet with round glasses, while a man in a purple shirt and white striped pants talks to them.
- 目标 prompt：A children's puppet show in a colorful room with bow-patterned wallpaper. A girl with a black bob haircut wearing a blue dress stands beside a brown cat puppet with round glasses, while a man in a purple shirt and white striped pants talks to them.

### 玩偶 黄裙 → 印花裙，去掉 cup：A / C / M / MR

- 编辑词：a yellow dress → a dress with a bold floral pattern（SAM3：yellow dress）
- 源 prompt：A children's puppet show in a colorful room with bow-patterned wallpaper. A girl with a black bob haircut wearing a yellow dress stands beside a brown cat puppet with round glasses, while a man in a purple shirt and white striped pants talks to them.
- 目标 prompt：A children's puppet show in a colorful room with bow-patterned wallpaper. A girl with a black bob haircut wearing a dress with a bold floral pattern stands beside a brown cat puppet with round glasses, while a man in a purple shirt and white striped pants talks to them.

### 自行车 红车 → 木头车架：memcheck C / M

- 编辑词：a red fixed-gear bicycle → a fixed-gear bicycle with a wooden frame（SAM3：bicycle）
- 源 prompt：A young man rides a red fixed-gear bicycle along a paved path lined with green lawns and tall trees on a sunny evening. He wears a black t-shirt, dark jeans and grey sneakers.
- 目标 prompt：A young man rides a fixed-gear bicycle with a wooden frame along a paved path lined with green lawns and tall trees on a sunny evening. He wears a black t-shirt, dark jeans and grey sneakers.

### 卡通 粉蝴蝶结 → 绿底白点蝴蝶结：memcheck C / M

- 编辑词：a pink bow → a large green bow with white polka dots（SAM3：pink bow）
- 源 prompt：A 2D cartoon scene. A girl with blonde pigtails, a pink bow and a pink dress stands with a boy with a football-shaped head, a blue shirt and a plaid kilt, in front of brick buildings on a city street.
- 目标 prompt：A 2D cartoon scene. A girl with blonde pigtails, a large green bow with white polka dots and a pink dress stands with a boy with a football-shaped head, a blue shirt and a plaid kilt, in front of brick buildings on a city street.

### 女士 黑白衬衫 → 绿色针织毛衣：memcheck C / M

- 编辑词：a black and white patterned blouse → a chunky green knitted sweater（SAM3：blouse）
- 源 prompt：A woman with dark curly hair and glasses, wearing a black and white patterned blouse and a party hat, tidies up a living room scattered with colorful balloons, with a sofa and a television.
- 目标 prompt：A woman with dark curly hair and glasses, wearing a chunky green knitted sweater and a party hat, tidies up a living room scattered with colorful balloons, with a sofa and a television.

