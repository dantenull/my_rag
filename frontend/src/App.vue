<script setup>
import Menus from './components/Menus.vue'

</script>

<template>
    <el-container>
        <el-aside>
            <Menus />
        </el-aside>
        <el-main>
            <router-view>

            </router-view>
        </el-main>
    </el-container>
</template>

<!-- <style scoped>
.layout-container-demo .el-aside {
  color: var(--el-text-color-primary);
  background: var(--el-color-primary-light-8);
}
.layout-container-demo .el-main {
  padding: 0;
}
</style> -->
